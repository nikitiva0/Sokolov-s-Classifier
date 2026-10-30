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
            for metric, value in (result['metrics'] or {}).items():
                table.append({'model': name, 'level': level, 'metric': metric, 'value': value,
                              'undefined_reason': result.get('undefined', {}).get(metric, '')})
            for cls, values in result.get('classes', {}).items():
                for metric, value in values.items():
                    table.append({'model': name, 'level': level, 'metric': cls+'.'+metric,
                                  'value': value, 'undefined_reason': result['undefined'].get(cls+'.'+metric, '')})
    report = run/'report'
    report.mkdir(exist_ok=True)
    audit = read_json(run/'data/data_audit.json')
    write_json(report/'metrics.json', summaries)
    write_json(report/'data_audit_summary.json', {k: v for k, v in audit.items()
               if k not in ('exclusions', 'near_duplicate_candidates')})
    write_csv(report/'metrics.csv', table)
    title = '# Соколов / не Соколов — техническая проверка' if technical else '# Соколов / не Соколов — сравнение моделей'
    lines = [title, '', 'Это технический прогон со случайными весами, не результат качества моделей.'
             if technical else 'Результаты полного обучения на предобученных моделях.', '',
             'Метки: 1 — Соколов, 0 — не Соколов. Пороги выбраны только на validation.', '',
             '## Ограничения', '']
    lines.extend('- '+warning for warning in audit['warnings'])
    lines += ['', '## Данные', '', '```json', json.dumps(audit['counts'], ensure_ascii=False, indent=2), '```', '',
              f"Исключено на аудите: {len(audit['exclusions'])}. Кандидатов pHash: {len(audit['near_duplicate_candidates'])}.",
              f"Время детекции: {audit['detection_seconds']:.2f} с. Детектор: {audit['detector']['name']}.", '',
              '## Метрики', '', '| Модель | Уровень | Метрика | Значение |', '|---|---|---|---|']
    for row in table:
        value = f"{row['value']:.6g}" if row['value'] is not None else 'N/A: '+row['undefined_reason']
        lines.append(f"| {row['model']} | {row['level']} | {row['metric']} | {value} |")
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
        destination = report/'synthesis'
        destination.mkdir(exist_ok=True)
        write_json(destination/'summary.json', synthesis_summary)
        lines += ['', '## Генерация из случайного шума', '',
                  'Веса выбранной модели фиксированы. Adam изменяет только пиксели входной картинки.',
                  'Максимизируется логит «Соколов» с TV- и L2-регуляризацией. После каждого шага пиксели ограничены диапазоном [0,1].',
                  'Оценки на графике рассчитаны по сохранённым 8-битным пикселям. Детектор лиц в оптимизации не участвует.',
                  'Высокая оценка не гарантирует узнаваемое лицо. Показаны все начальные шумы; рост оценки не обязан быть монотонным.', '',
                  '```json', json.dumps(synthesis_summary, ensure_ascii=False, indent=2), '```', '']
        for image in ('seed_comparison.png', 'scores.png'):
            shutil.copy2(run/'synthesis'/image, destination/image)
            lines.append(f'![Synthesis {image}](synthesis/{image})')
        for seed in synthesis_summary['seeds']:
            target = destination/f'seed_{seed}'
            target.mkdir(exist_ok=True)
            for name in ('progression.png', 'trajectory.csv'):
                shutil.copy2(run/'synthesis'/f'seed_{seed}'/name, target/name)
            lines += ['', f'### Начальный шум {seed}', '',
                      f'![Steps for seed {seed}](synthesis/seed_{seed}/progression.png)', '']
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
        allowed |= {'synthesis/summary.json', 'synthesis/seed_comparison.png', 'synthesis/scores.png'}
        allowed |= {f'synthesis/seed_{seed}/{name}' for seed in seeds
                    for name in ('progression.png', 'trajectory.csv')}
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
