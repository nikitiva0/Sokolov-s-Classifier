"""Small localhost-only image upload UI; files stay in memory."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
from urllib.parse import urlsplit

from .inference import MAX_UPLOAD_BYTES, PhotoClassifier


def create_server(classifier, port=8766):
    token = secrets.token_urlsafe(32)
    page = (Path(__file__).parent/'web_ui.html').read_text().replace('__REQUEST_TOKEN__', token)

    class Handler(BaseHTTPRequestHandler):
        def send(self, code, body, kind='application/json; charset=utf-8'):
            if isinstance(body, dict):
                body = json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(code)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.send_header('Content-Security-Policy', "default-src 'self'; img-src 'self' data: blob:; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if urlsplit(self.path).path == '/':
                self.send(200, page.encode(), 'text/html; charset=utf-8')
            elif self.path == '/api/status':
                self.send(200, classifier.info())
            else:
                self.send(404, {'error': 'Страница не найдена.'})

        def do_POST(self):
            if self.path != '/api/predict':
                self.send(404, {'error': 'Страница не найдена.'})
                return
            if not secrets.compare_digest(self.headers.get('X-Request-Token', ''), token):
                self.send(403, {'error': 'Обновите страницу и повторите проверку.'})
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
            except ValueError:
                length = 0
            if not 0 < length <= MAX_UPLOAD_BYTES:
                self.close_connection = True
                self.send(413, {'error': 'Выберите изображение размером до 15 МБ.'})
                return
            self.connection.settimeout(30)
            try:
                data = self.rfile.read(length)
                if len(data) != length:
                    raise ValueError('Загрузка прервалась. Выберите файл ещё раз.')
                self.send(200, classifier.classify(data))
            except ValueError as error:
                self.send(400, {'error': str(error)})
            except TimeoutError:
                self.close_connection = True
                self.send(408, {'error': 'Загрузка заняла слишком много времени. Повторите попытку.'})
            except Exception:
                self.send(500, {'error': 'Не удалось проверить фото. Попробуйте другое изображение.'})

        def log_message(self, *_):
            pass

    return ThreadingHTTPServer(('127.0.0.1', port), Handler)


def serve(run_dir, *, port=8766, device='auto', detector_root=None):
    import os
    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '0')
    print('Загружаю обученную модель и детектор…', flush=True)
    classifier = PhotoClassifier(run_dir, device=device, detector_root=detector_root)
    server = create_server(classifier, port)
    print(f'Откройте http://127.0.0.1:{server.server_port} · {classifier.architecture} · {classifier.device}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
