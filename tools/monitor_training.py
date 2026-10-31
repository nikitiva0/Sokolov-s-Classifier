"""Read-only local dashboard for the actual training run; binds localhost only."""
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import argparse,csv,json,time,re

ROOT=Path(__file__).resolve().parents[1]

def read(path,default=None):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return default

def csv_rows(path):
    try:
        with path.open() as stream:return list(csv.DictReader(stream))
    except OSError:return []

def status(run):
    runtime=read(ROOT/'work/local-mps/notebook-status.json',{})
    run=Path(runtime.get('run_dir',run))
    arc=ROOT/'checkpoints/arcface/iresnet50_ms1mv3.pt'
    detector=ROOT/'checkpoints/insightface/models/buffalo_l/det_10g.onnx'
    fairface=ROOT/'work/local-mps/fairface_path.txt'
    resnet=ROOT/'work/torch-cache/hub/checkpoints/resnet18-f37072fd.pth'
    assets=[{'name':'ResNet18 · ImageNet','ready':resnet.exists(),'bytes':resnet.stat().st_size if resnet.exists() else 0},
        {'name':'ArcFace · MS1MV3','ready':arc.exists() and (arc.parent/'source.json').exists(),'bytes':arc.stat().st_size if arc.exists() else 0},
        {'name':'Детектор лиц SCRFD','ready':detector.exists(),'bytes':detector.stat().st_size if detector.exists() else 0},
        {'name':'FairFace · негативные фото','ready':fairface.exists(),'bytes':None}]
    models=[]
    for name,total in [('resnet18',15),('arcface',5)]:
        folder=run/name
        state=read(folder/'status.json',{})
        progress=read(folder/'progress.json',{})
        history=csv_rows(folder/'training.csv')
        models.append({'id':name,'total_epochs':total,'state':state.get('status','queued'),
            'progress':progress,'history':history,'summary':read(folder/'training_summary.json')})
    prepared_folder=Path(runtime.get('prepared_manifest',str(ROOT/'data/processed/local_mps/manifest.csv'))).parent
    prepared=read(prepared_folder/'prepare_progress.json',{})
    if (prepared_folder/'prepared.json').exists():prepared['completed']=True
    snapshots=list((run/'synthesis').glob('seed_*/completed.json')) if (run/'synthesis').exists() else []
    return {'updated':time.time(),'run_id':run.name,'runtime':runtime,'assets':assets,'models':models,
        'prepare':prepared,'generated_seeds':len(snapshots),'seed_count':5,
        'complete':(run/'report/completed.json').exists(),'run_status':read(run/'status.json',{}),
        'report_available':(run/'report/report.html').exists(),'device':'mps','hardware':'Apple M4 · 16 ГБ'}

PAGE='''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Соколов · прогресс обучения</title>
<style>
:root{color-scheme:dark;font-family:system-ui,-apple-system,sans-serif;background:#0c111b;color:#e8edf7}*{box-sizing:border-box}body{margin:0;padding:32px;max-width:1240px;margin:auto}header{display:flex;justify-content:space-between;gap:16px;align-items:center;margin-bottom:28px}h1{font-size:25px;margin:0 0 8px;font-weight:600}h2{font-size:18px;margin:0 0 16px}p{color:#a6b2c8;margin:6px 0;line-height:1.5}.badge{padding:8px 12px;background:#122a31;color:#8fe6d9;border-radius:8px;white-space:nowrap}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:16px;margin:20px 0}.card{background:#151d2b;border:1px solid #273247;border-radius:14px;padding:22px}.big{font-size:27px;font-weight:600;margin:8px 0}.small{font-size:13px;color:#a6b2c8}.steps{display:grid;grid-template-columns:repeat(7,1fr);gap:12px;margin:26px 0}.step{border-top:3px solid #29364a;padding:13px 0;color:#8d9ab0;font-size:14px}.step.active{border-color:#59c9d0;color:#e8edf7}.step.done{border-color:#6dd69f;color:#c2ead2}.step.error{border-color:#ef8585;color:#efaaaa}.status{background:#14283a;padding:18px 22px;border-radius:12px;margin:20px 0;line-height:1.6}.row{display:flex;justify-content:space-between;gap:12px;margin:12px 0;align-items:center}.ok{color:#85dfae}.wait{color:#b5bfd0}.bar{height:7px;border-radius:5px;background:#293449;overflow:hidden;margin:14px 0}.fill{height:100%;background:#58c4cc;transition:width .4s}table{width:100%;border-collapse:collapse;font-size:14px}td,th{padding:10px 4px;border-bottom:1px solid #293449;text-align:left;font-weight:400}th{color:#9facc2}td:last-child{text-align:right}.errorText{color:#ffaaaa}a{color:#83dbe4}svg{width:100%;height:110px}footer{color:#8494ae;font-size:12px;margin-top:24px}@media(max-width:750px){body{padding:18px}.steps{grid-template-columns:repeat(2,1fr)}header{align-items:flex-start;flex-direction:column}.cards{grid-template-columns:1fr}}
</style>
<header><div><h1>Соколов · обучение моделей</h1><p id="run">Локальный запуск</p></div><span class="badge">Apple M4 · строго MPS</span></header>
<div class="status"><strong id="stage">Подключаюсь к запуску…</strong><p id="detail">Читаю фактические результаты.</p></div>
<div class="steps" id="steps"></div>
<div class="cards"><section class="card"><span class="small">Ускорение обучения</span><div class="big">MPS</div><p>Apple Metal · без CPU fallback</p></section><section class="card"><span class="small">Выборка перед аудитом</span><div class="big">1 000 + 5 000</div><p>Соколов + FairFace</p></section><section class="card"><span class="small">Итоговый отчёт</span><div class="big" id="report">Ожидает обучения</div><p id="reportlink">Метрики и генерация из пяти шумов</p></section></div>
<div class="cards"><section class="card"><h2>Данные и предобученные веса</h2><div id="assets"></div><p id="prepare" class="small"></p></section><section class="card" id="model0"></section><section class="card" id="model1"></section></div>
<footer id="refresh">Обновление каждые 2 секунды. Проценты появляются только там, где известно фактическое число обработанных объектов.</footer>
<script>
const names=['Данные и веса','Подготовка лиц','ResNet18','ArcFace','Генерация','Отчёт','Android-экспорт'];
const fmt=v=>Number.isFinite(Number(v))?Number(v).toFixed(4):'—';
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function chart(h){if(h.length<2)return '<p class="small">Кривая качества появится после первых эпох.</p>';let points=h.map((r,i)=>`${12+i*276/Math.max(1,h.length-1)},${98-Number(r.validation_ap)*84}`).join(' ');return `<svg viewBox="0 0 300 110" aria-label="Validation AP по эпохам"><path d="M12 10V98H288" fill="none" stroke="#3b4961"/><polyline points="${points}" fill="none" stroke="#58c4cc" stroke-width="2"/></svg>`;}
function render(s){document.getElementById('run').textContent=s.run_id+' · '+s.hardware;let stage=s.runtime.stage||'Загрузка данных и весов';document.getElementById('stage').textContent=stage;document.getElementById('detail').textContent=s.runtime.error||s.runtime.detail||'Обучение ещё не началось. Загружаются FairFace и необходимые модели.';
let allAssets=s.assets.every(a=>a.ready), prep=s.prepare.completed, m=s.models, current=s.runtime.stage_index??0;
const done=[allAssets,prep,...m.map(x=>['trained','evaluated'].includes(x.state)),s.generated_seeds===s.seed_count,s.complete,s.runtime.mobile_exported===true];
document.getElementById('steps').innerHTML=names.map((n,i)=>`<div class="step ${done[i]?'done':i===current?(s.runtime.error?'error':'active'):''}">${done[i]?'✓':String(i+1).padStart(2,'0')} ${n}</div>`).join('');
document.getElementById('assets').innerHTML=s.assets.map(a=>`<div class="row"><span>${esc(a.name)}</span><span class="${a.ready?'ok':'wait'}">${a.ready?'✓ Готово':'Загрузка…'}</span></div>`).join('');
document.getElementById('prepare').textContent=s.prepare.total?`Обработано кадров: ${s.prepare.processed} / ${s.prepare.total}`:'Поиск и выравнивание лиц начнутся после загрузки.';
m.forEach((model,i)=>{let p=model.progress,h=model.history,epoch=p.epoch||h.length, batch=p.batch||0,total=p.batches||0,percent=['trained','evaluated'].includes(model.state)?100:(p.overall_percent||0),last=h.at(-1);document.getElementById('model'+i).innerHTML=`<h2>${model.id==='resnet18'?'ResNet18':'ArcFace · iResNet50'}</h2><span class="small">${model.state==='queued'?'Ожидает запуска':model.state==='training'?'Обучение на MPS':'Обучение завершено'}</span><div class="big">${epoch} / ${model.total_epochs} эпох</div><div class="bar"><div class="fill" style="width:${percent}%"></div></div><p class="small">${total?`Батч ${batch} / ${total}`:'Батчи ещё не обрабатываются'}</p><table><tr><th>Loss</th><td>${fmt(p.loss??last?.train_loss)}</td></tr><tr><th>Validation AP</th><td>${fmt(last?.validation_ap)}</td></tr><tr><th>Завершено эпох</th><td>${h.length}</td></tr></table>${chart(h)}`;});
document.getElementById('report').textContent=s.complete?'Готов':`${s.generated_seeds} / 5 шумов`;
if(s.report_available)document.getElementById('reportlink').innerHTML='<a href="/report" target="_blank">Открыть отчёт</a>';
document.getElementById('refresh').textContent='Обновлено: '+new Date(s.updated*1000).toLocaleTimeString('ru-RU')+' · Реальные данные запуска · Без прогнозных процентов';}
async function poll(){try{let r=await fetch('/status',{cache:'no-store'});if(!r.ok)throw Error(r.status);render(await r.json());}catch(e){document.getElementById('detail').textContent='Связь с локальным монитором потеряна. Обновление повторится автоматически.';}}
poll();setInterval(poll,2000);
</script></html>'''

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--port',type=int,default=8765)
    args=parser.parse_args();run=args.run_dir.resolve()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path=='/status':content=json.dumps(status(run),ensure_ascii=False).encode();kind='application/json'
            elif self.path=='/':content=PAGE.encode();kind='text/html; charset=utf-8'
            elif self.path=='/report' and (Path(read(ROOT/'work/local-mps/notebook-status.json',{}).get('run_dir',run))/'report/report.html').exists():content=(Path(read(ROOT/'work/local-mps/notebook-status.json',{}).get('run_dir',run))/'report/report.html').read_bytes();kind='text/html; charset=utf-8'
            else:self.send_error(404);return
            self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(content)
        def log_message(self,*args):pass
    print(f'http://127.0.0.1:{args.port}',flush=True)
    ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()
