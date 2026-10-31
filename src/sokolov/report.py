"""Comparative, self-contained local reports and narrowly scoped Git publication."""
from __future__ import annotations
import base64
import html
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from .io import read_csv, read_json, sha256, write_csv, write_json

MODELS = ('resnet18', 'arcface')
LOCAL_AUDIT_KEYS = ('exclusions', 'near_duplicate_candidates', 'near_duplicate_excluded')


def _fmt(value, digits=3):
    return 'N/A' if value is None else f'{value:.{digits}f}'


def protocol_lines(audit):
    """Markdown description of the split protocol and the near-duplicate guard."""
    summary, guard = audit.get('split_summary') or {}, audit.get('near_duplicate_guard') or {}
    mode = audit.get('split_mode', summary.get('mode'))
    lines = ['## Протокол оценки', '']
    if mode == 'group_stratified':
        checks = summary['checks']
        lines += [
            'Единица разбиения — группа: лекция или видео с Соколовым (вместе с посторонними в кадре), '
            'известный другой человек (все его видео), отдельный файл FairFace. Каждая группа целиком попадает '
            'ровно в одну выборку: кадры одной лекции не делятся между train, validation и test.',
            f"Группы стратифицированы по паре (метка группы, домен): около {summary['validation_fraction']:.0%} групп "
            f"каждого страта — в validation, {summary['test_fraction']:.0%} — в test, seed={summary['seed']}. "
            'Гарантии: в каждой выборке оба класса; в test есть лекция из набора v1 и web-источники Соколова из '
            'разных доменов (если они есть); в validation есть позитивная группа.',
            f"Проверка: групп сразу в нескольких выборках — {len(checks['groups_in_several_splits'])}; "
            f"лекций v1 в test — {len(checks['test_v1_lectures'])}; web-доменов Соколова в test — "
            f"{', '.join(checks['test_web_positive_domains']) or 'нет'} "
            f"(доступно: {', '.join(checks['web_positive_domains_available']) or 'нет'}); "
            f"позитивных групп в validation — {checks['val_positive_groups']}.", '']
    elif mode == 'random_stratified':
        lines += ['Случайное стратифицированное разбиение отдельных кадров (явно выбранный устаревший режим). '
                  'Кадры одной лекции могут попасть в разные выборки; метрики test завышены.', '']
    else:
        lines += ['Внешнее разбиение train/test сохранено; validation выделена из train по группам.', '']
    lines.append('Порог решения и лучший checkpoint выбираются только по validation; test используется один раз.')
    if guard:
        excluded = guard['excluded']
        lines += ['', f"Защита от почти-дубликатов: для каждого выровненного лица считается 64-битный pHash. Лица "
                  f"validation и test на расстоянии Хэмминга ≤ {guard['max_distance']} бит от любого лица train, а также "
                  'лица test рядом с лицами validation получают статус near_duplicate и исключаются из оценки. '
                  f"Исключено: validation — {sum(excluded['val'].values())} (Соколов: {excluded['val']['1']}), "
                  f"test — {sum(excluded['test'].values())} (Соколов: {excluded['test']['1']}).",
                  f"Минимальное межвыборочное расстояние, бит, до фильтра: {json.dumps(guard['min_cross_split_distance_before'])}; "
                  f"после: {json.dumps(guard['min_cross_split_distance_after'])}."]
    if summary.get('cells'):
        lines += ['', '| Выборка | Метка | Домен | Групп | Изображений | Лиц ok | Почти-дубликатов |',
                  '|---|---|---|---|---|---|---|']
        for c in summary['cells']:
            lines.append(f"| {c['split']} | {c['label']} | {c['domain'] or 'unknown'} | {c['groups']} | {c['images']} "
                         f"| {c.get('faces_ok', 'N/A')} | {c.get('near_duplicate', 'N/A')} |")
    return lines + ['']


def breakdown_lines(name, evaluation):
    """Per-domain and per-source test diagnostics; no faces, only aggregates."""
    result = evaluation.get('breakdown')
    if not result:
        return []
    lines = ['', f'### {name}: разбивка test по доменам и источникам', '',
             '| Домен | Позитивов | Негативов | Recall | FPR | ROC-AUC |', '|---|---|---|---|---|---|']
    for domain, v in result['by_domain'].items():
        lines.append(f"| {domain} | {v['positives']} | {v['negatives']} | {_fmt(v['recall'])} | "
                     f"{_fmt(v['false_positive_rate'])} | {_fmt(v['roc_auc'])} |")
    lines += ['', '| Источник Соколова | Домен | Изображений | Найдено лиц | Recall | Медиана оценки |',
              '|---|---|---|---|---|---|']
    for source, v in result['positive_sources'].items():
        source = source.replace('|', '\\|')
        lines.append(f"| {source} | {v['domain']} | {v['positives']} | {v['detected']} | {_fmt(v['recall'])} | "
                     f"{_fmt(v['median_score'])} |")
    lines += ['', '| Негативы | Изображений | Ложных срабатываний | FPR |', '|---|---|---|---|']
    names = {'in_domain': 'сложные негативы из тех же условий съёмки', 'fairface': 'FairFace'}
    for kind, v in result['negatives_by_kind'].items():
        lines.append(f"| {names.get(kind, kind)} | {v['negatives']} | {v['false_positives']} | "
                     f"{_fmt(v['false_positive_rate'])} |")
    return lines + ['']


def make_report(run_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    run = Path(run_dir).resolve()
    summaries, table, technical = {}, [], False
    for name in MODELS:
        folder = run/name
        status = read_json(folder/'status.json')
        if status['status'] != 'evaluated':
            raise ValueError(f'Cannot report an unfinished model: {name}')
        metrics = read_json(folder/'test_metrics.json')
        summary = read_json(folder/'training_summary.json')
        technical |= status['technical_test']
        if summary['data_fingerprint'] != read_json(run/MODELS[0]/'training_summary.json')['data_fingerprint']:
            raise ValueError('Models used different data/splits')
        summaries[name] = {'evaluation': metrics, 'training': summary,
                           'environment': read_json(folder/'environment.json')}
        for level in ('image_level', 'face_level'):
            result = metrics[level]
            intervals = ((metrics.get('bootstrap') or {}).get(level) or {}).get('intervals') or {}
            for metric, value in (result['metrics'] or {}).items():
                ci = intervals.get(metric) or {}
                table.append({'model': name, 'level': level, 'metric': metric, 'value': value,
                              'undefined_reason': result.get('undefined', {}).get(metric, ''),
                              'ci95_low': ci.get('low', ''), 'ci95_high': ci.get('high', '')})
            for cls, values in result.get('classes', {}).items():
                for metric, value in values.items():
                    table.append({'model': name, 'level': level, 'metric': cls+'.'+metric,
                                  'value': value, 'undefined_reason': result['undefined'].get(cls+'.'+metric, ''),
                                  'ci95_low': '', 'ci95_high': ''})
    report = run/'report'
    report.mkdir(exist_ok=True)
    audit = read_json(run/'data/data_audit.json')
    write_json(report/'metrics.json', summaries)
    write_json(report/'data_audit_summary.json', {k: v for k, v in audit.items() if k not in LOCAL_AUDIT_KEYS})
    write_csv(report/'metrics.csv', table)
    title = '# Соколов / не Соколов — техническая проверка' if technical else '# Соколов / не Соколов — сравнение моделей'
    lines = [title, '', 'Это технический прогон со случайными весами, не результат качества моделей.'
             if technical else 'Результаты полного обучения на предобученных моделях.', '',
             'Метки: 1 — Соколов, 0 — не Соколов. Пороги выбраны только на validation.', '']
    lines += protocol_lines(audit)
    lines += ['## Ограничения', '']
    lines.extend('- '+warning for warning in audit['warnings'])
    bootstrap = (summaries[MODELS[0]]['evaluation'].get('bootstrap') or {}).get('image_level') or {}
    candidates = audit.get('near_duplicate_candidate_counts', {}).get('cross_group_pairs',
                                                                      len(audit['near_duplicate_candidates']))
    lines += ['', '## Данные', '', '```json', json.dumps(audit['counts'], ensure_ascii=False, indent=2), '```', '',
              f"Исключено на аудите: {len(audit['exclusions'])}. Пар исходников разных групп с близким pHash: {candidates}. "
              f"Правило нескольких лиц: {audit.get('primary_face_rule', 'единственное лицо')}.",
              f"Время детекции: {audit['detection_seconds']:.2f} с. Детектор: {audit['detector']['name']}.", '',
              '## Метрики', '',
              '95% ДИ — групповой бутстрэп (целые группы с возвращением, стратификация по метке группы, '
              f"{bootstrap.get('resamples', 'N/A')} повторов, seed={bootstrap.get('seed', 'N/A')}) "
              'для ROC-AUC, AP и balanced accuracy.', '',
              '| Модель | Уровень | Метрика | Значение | 95% ДИ |', '|---|---|---|---|---|']
    for row in table:
        value = f"{row['value']:.6g}" if row['value'] is not None else 'N/A: '+row['undefined_reason']
        interval = f"{row['ci95_low']:.4g}–{row['ci95_high']:.4g}" if row['ci95_low'] != '' else ''
        lines.append(f"| {row['model']} | {row['level']} | {row['metric']} | {value} | {interval} |")
    for name in MODELS:
        folder = run/name
        plots = report/name
        plots.mkdir(exist_ok=True)
        for path in (folder/'plots').glob('*.png'):
            shutil.copy2(path, plots/path.name)
        history = read_csv(folder/'training.csv')
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot([int(r['epoch']) for r in history], [float(r['train_loss']) for r in history])
        axes[0].set(xlabel='Epoch', ylabel='Train loss')
        axes[1].plot([int(r['epoch']) for r in history], [float(r['validation_ap']) for r in history])
        axes[1].set(xlabel='Epoch', ylabel='Validation AP')
        fig.tight_layout(); fig.savefig(plots/'training.png', dpi=140); plt.close(fig)
        summary = summaries[name]
        lines += ['', f'## {name}', '', '```json', json.dumps({
            'training_seconds': summary['training']['training_seconds'],
            'parameters': summary['training']['parameters'],
            'checkpoint_bytes': summary['evaluation']['checkpoint_bytes'],
            'benchmark': summary['evaluation']['benchmark'],
            'coverage': summary['evaluation']['coverage'],
            'pretrained_weights': summary['training']['pretrained_provenance'],
            'environment': summary['environment'],
            'threshold': read_json(folder/'threshold.json')}, ensure_ascii=False, indent=2), '```', '']
        for image in ['training.png', 'curves.png', 'confusion_matrix.png', 'scores.png']:
            lines.append(f'![{name} {image}]({name}/{image})')
        lines += breakdown_lines(name, summary['evaluation'])
        errors = read_csv(folder/'errors.csv')
        lines += ['', 'Примеры ошибок (полный список и исходные пути — в локальном errors.csv):', '',
                  '| ID | Истинная метка | Оценка | Тип |', '|---|---|---|---|']
        for row in errors[:20]:
            image_id = row['image_id'].replace('|', '\\|').replace('\n', ' ')
            lines.append(f"| {image_id} | {row['label']} | {float(row['score']):.5f} | {row['error_type']} |")
        # Local-only visual examples; never copied by publication allowlist.
        from PIL import Image, ImageDraw
        sheet = Image.new('RGB', (5*160, 4*150), 'white')
        draw = ImageDraw.Draw(sheet)
        for i, row in enumerate(errors[:20]):
            with Image.open(row['path']) as im:
                im = im.convert('RGB'); im.thumbnail((160, 116))
                x, y = i % 5 * 160, i // 5 * 150
                sheet.paste(im, (x, y)); draw.text((x, y+118), f"{row['error_type']} {float(row['score']):.3f}", fill='black')
        sheet.save(folder/'error_examples.jpg')
    synthesis_summary = None
    if (run/'best_model.json').exists():
        winner = read_json(run/'best_model.json')
        write_json(report/'best_model.json', winner)
        lines += ['', '## Лучшая сохранённая модель', '',
                  f"Победитель: **{winner['architecture']}**, validation AP={winner['validation_ap']:.6f}, эпоха {winner['epoch']}.",
                  'Выбор выполнен по validation, без использования test. Полный checkpoint сохранён локально в best_model.pt.', '']
    if (run/'synthesis').exists():
        from .synthesize import validate_synthesis
        synthesis_summary = validate_synthesis(run/'synthesis')
        if synthesis_summary['technical_test'] != technical:
            raise ValueError('Synthesis and classifier reports have different technical-test status')
        winner = read_json(run/'best_model.json')
        if synthesis_summary['checkpoint_sha256'] != winner['checkpoint_sha256']:
            raise ValueError('Synthesis used a different winner checkpoint')
        from .analysis import report_section
        lines += report_section(run, report/'synthesis', synthesis_summary)
    else:
        lines += ['', '## Генерация', '', 'В этом отчёте генерация ещё не запускалась.', '']
    lines += ['', '## Воспроизводимость', '',
              'FairFace: https://github.com/joojs/fairface (CC BY 4.0).',
              'Параметры, состояния обучения, исходные предсказания и аудит сохранены в файлах запуска.',
              'Веса InsightFace предназначены для некоммерческих исследований.', '']
    (report/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    import markdown
    rendered = markdown.markdown('\n'.join(lines), extensions=['tables', 'fenced_code'])
    def embed(match):
        path = report/match.group(1)
        return 'src="data:image/png;base64,' + base64.b64encode(path.read_bytes()).decode() + '"'
    rendered = re.sub(r'src="([^":]+\.png)"', embed, rendered)
    (report/'report.html').write_text('<!doctype html><html lang="ru"><meta charset="utf-8">'
        '<title>Sokolov classifier</title><style>body{max-width:1100px;margin:40px auto;font:16px system-ui;padding:20px}'
        'table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:6px}img{max-width:100%}'
        'pre{overflow:auto;background:#f3f4f6;padding:12px}</style>'+rendered+'</html>', encoding='utf-8')
    # A completion manifest binds only publication-safe artifacts, no weights/datasets/secrets.
    manifest = {p.relative_to(report).as_posix(): sha256(p) for p in sorted(report.rglob('*'))
                if p.is_file() and p.name != 'completed.json'}
    write_json(report/'completed.json', {'status': 'completed', 'technical_test': technical,
        'models': list(MODELS), 'synthesis': synthesis_summary,
        'best_model': (report/'best_model.json').exists(), 'files': manifest})
    return report


def validate_publication(run_dir):
    report = Path(run_dir)/'report'
    completed = read_json(report/'completed.json')
    if completed['status'] != 'completed' or completed['technical_test']:
        raise ValueError('Only a completed real pretrained-model run can be published')
    allowed = {'report.md', 'report.html', 'metrics.json', 'metrics.csv', 'data_audit_summary.json'}
    allowed |= {f'{name}/{plot}.png' for name in MODELS for plot in ('training', 'curves', 'confusion_matrix', 'scores')}
    if completed.get('best_model'):
        allowed.add('best_model.json')
    synthesis = completed.get('synthesis')
    if synthesis is not None:
        seeds = synthesis['seeds']
        if (not seeds or len(seeds) != len(set(seeds)) or
                any(type(s) is not int or s < 0 for s in seeds) or synthesis['technical_test']):
            raise ValueError('Invalid completed synthesis experiment')
        from .analysis import publication_files
        allowed |= publication_files(synthesis)
    if set(completed['files']) != allowed or completed['models'] != list(MODELS):
        raise ValueError('Unexpected publication artifact list')
    for relative, digest in completed['files'].items():
        path = report/relative
        if path.is_symlink() or sha256(path) != digest:
            raise ValueError('Publication artifact modified')
    return report, completed


def publish_report(run_dir, repository='https://github.com/nikitiva0/Sokolov-s-Classifier.git',
                   *, transport=None):
    report, completed = validate_publication(run_dir)
    run_id = Path(run_dir).resolve().name
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,100}', run_id):
        raise ValueError('Run directory needs a safe unique name')
    if transport:
        return transport(report, run_id, completed)
    # Use a separate checkout: never stage unrelated user changes or force push.
    import os
    with tempfile.TemporaryDirectory(prefix='sokolov-publish-') as tmp:
        checkout = Path(tmp)/'repo'
        env = dict(os.environ)
        token = env.pop('SOKOLOV_GITHUB_TOKEN', '')
        if token:
            auth = base64.b64encode(('x-access-token:'+token).encode()).decode()
            env.update(GIT_CONFIG_COUNT='1', GIT_CONFIG_KEY_0='http.https://github.com/.extraheader',
                       GIT_CONFIG_VALUE_0='AUTHORIZATION: basic '+auth)
        def git(*args, cwd=None):
            # Git messages may contain credential-bearing URLs: do not echo them on failure.
            result = subprocess.run(['git', *args], cwd=cwd, env=env, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError('Git publication failed; local report is preserved. Check authentication and remote branch.')
            return result.stdout.strip()
        git('clone', '--depth', '1', '--branch', 'main', repository, str(checkout))
        destination = checkout/'reports'/run_id
        if destination.exists():
            raise ValueError('Report run_id already published; choose a new run directory')
        destination.mkdir(parents=True)
        for relative in completed['files']:
            target = destination/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(report/relative, target)
        shutil.copy2(report/'completed.json', destination/'completed.json')
        git('add', '--', f'reports/{run_id}', cwd=checkout)
        git('-c', 'user.name=Sokolov report publisher', '-c', 'user.email=sokolov-report@users.noreply.github.com',
            'commit', '-m', f'Publish classifier report {run_id}', cwd=checkout)
        git('push', 'origin', 'HEAD:main', cwd=checkout)
        return git('rev-parse', 'HEAD', cwd=checkout)
