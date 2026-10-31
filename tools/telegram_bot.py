"""Telegram-бот учебного классификатора: Bot API через стандартную библиотеку.

Long polling (getUpdates), по одному обновлению за раз. Фотографии обрабатываются
только в памяти тем же PhotoClassifier, что и локальный UI (SCRFD, выравнивание,
победитель и порог из validation). Токен не печатается и вырезается из ошибок.

    PYTHONPATH=src python tools/telegram_bot.py --run-dir outputs/runs/now-v3-eq --device mps
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import http.client
from io import BytesIO
import json
import os
from pathlib import Path
import re
import signal
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOKEN_FILE = ROOT/'work'/'telegram_token'
DEFAULT_API_BASE = 'https://api.telegram.org'
MAX_FILE_BYTES = 10*1024*1024
MAX_API_RESPONSE = 4*1024*1024
MAX_FACE_LINES = 25
CAPTION_LIMIT = 1024
FATAL_CODES = (401, 404)
TOKEN_RE = re.compile(r'^[0-9]{5,}:[A-Za-z0-9_-]{30,}$')
COLORS = {'positive': (34, 177, 76), 'negative': (220, 40, 40), 'error': (235, 150, 20)}
VERDICTS = {
    'sokolov': 'На фото есть Соколов',
    'not_sokolov': 'Соколова на фото нет',
    'no_face': 'Лицо не найдено',
    'uncertain': 'Соколов не найден, но не все лица удалось проверить',
}
HELP = (
    'Учебный классификатор проектной школы НИУ ВШЭ: проверяет, есть ли на фотографии '
    'Евгений Соколов.\n\n'
    'Пришлите фото — обычным снимком или файлом JPG, PNG, WebP до 10 МБ. Бот найдёт лица, '
    'оценит каждое моделью и ответит:\n'
    '• «На фото есть Соколов» — хотя бы одно лицо прошло порог;\n'
    '• «Соколова на фото нет» — ни одно лицо не прошло порог;\n'
    '• «Лицо не найдено».\n'
    'В ответ придёт фото с рамками: зелёная — Соколов, красная — не Соколов.\n\n'
    'Это учебная модель, она ошибается. Оценка модели — не вероятность и не доказательство '
    'личности.\n'
    'Фотографии обрабатываются в памяти и не сохраняются. В журнал пишутся только id чата, '
    'число лиц, вердикт и время обработки.'
)
NOT_IMAGE = 'Это не изображение. Пришлите фото JPG, PNG или WebP. Подробнее: /help'
HINT = 'Пришлите фотографию, и я проверю, есть ли на ней Соколов. Подробнее: /help'


def utc_now():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'


def log(message):
    print(f'{utc_now()} {message}', flush=True)


class TelegramError(Exception):
    """Ошибка Bot API или сети; текст уже не содержит токена."""

    def __init__(self, message, *, code=None, retry_after=None, network=False):
        super().__init__(message)
        self.code, self.retry_after, self.network = code, retry_after, network


class FileTooLarge(ValueError):
    def __init__(self, limit):
        super().__init__(f'Файл больше {limit/1024/1024:g} МБ. Пришлите изображение поменьше.')


def check_api_base(base):
    parts = urllib.parse.urlsplit(base)
    local = parts.hostname in ('127.0.0.1', 'localhost', '::1')
    if parts.scheme == 'https' or (parts.scheme == 'http' and local):
        return base.rstrip('/')
    raise ValueError('--api-base: нужен https:// (http:// допустим только для localhost).')


def load_token(token_file=None, env=None):
    """Явный --token-file > TELEGRAM_BOT_TOKEN > work/telegram_token. Возвращает (токен, источник)."""
    env = os.environ if env is None else env
    if token_file is None and env.get('TELEGRAM_BOT_TOKEN', '').strip():
        token, source = env['TELEGRAM_BOT_TOKEN'].strip(), 'TELEGRAM_BOT_TOKEN'
    else:
        path = Path(token_file or DEFAULT_TOKEN_FILE)
        source = str(path)
        try:
            token = path.read_text(encoding='utf-8').strip()
        except OSError:
            raise ValueError(f'Токен не найден: задайте TELEGRAM_BOT_TOKEN или создайте {path}.') from None
        if path.stat().st_mode & 0o077:
            log(f'Предупреждение: {path} доступен другим пользователям; выполните chmod 600.')
    if not TOKEN_RE.match(token):
        raise ValueError(f'Токен из {source} имеет неверный формат (ожидается 123456:ABC…).')
    return token, source


def parse_allowed(value):
    try:
        return frozenset(int(item) for item in value.replace(' ', '').split(',') if item)
    except ValueError:
        raise ValueError('TELEGRAM_ALLOWED_CHATS: нужны целые id чатов через запятую.') from None


def encode_multipart(fields, files):
    """multipart/form-data из стандартной библиотеки; files: name -> (filename, bytes, type)."""
    boundary = uuid.uuid4().hex
    while any(boundary.encode() in content for _, content, _ in files.values()):
        boundary = uuid.uuid4().hex
    chunks = []
    for name, value in fields.items():
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        chunks.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'
                      f'{text}\r\n'.encode())
    for name, (filename, content, content_type) in files.items():
        chunks += [f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; '
                   f'filename="{filename}"\r\nContent-Type: {content_type}\r\n\r\n'.encode(),
                   content, b'\r\n']
    chunks.append(f'--{boundary}--\r\n'.encode())
    return b''.join(chunks), f'multipart/form-data; boundary={boundary}'


class TelegramAPI:
    def __init__(self, token, base=DEFAULT_API_BASE):
        self._token = token
        self.base = check_api_base(base)

    def __repr__(self):
        return f'TelegramAPI({self.base!r})'

    def redact(self, text):
        return str(text).replace(self._token, '<TOKEN>')

    def _network_error(self, what, error):
        return TelegramError(f'{what}: сеть недоступна ({type(error).__name__}: {self.redact(error)})',
                             network=True)

    def call(self, method, params=None, *, files=None, timeout=30):
        params = {key: value for key, value in (params or {}).items() if value is not None}
        if files:
            body, content_type = encode_multipart(params, files)
        else:
            body, content_type = json.dumps(params, ensure_ascii=False).encode(), 'application/json'
        request = urllib.request.Request(f'{self.base}/bot{self._token}/{method}', data=body,
                                         headers={'Content-Type': content_type}, method='POST')
        status = 200
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read(MAX_API_RESPONSE)
        except urllib.error.HTTPError as error:
            status = error.code
            try:
                raw = error.read(MAX_API_RESPONSE)
            except (OSError, http.client.HTTPException):
                raw = b''
            finally:
                error.close()
        except (OSError, http.client.HTTPException) as error:
            raise self._network_error(method, error) from None
        try:
            payload = json.loads(raw)
        except ValueError:
            payload = None
        if not isinstance(payload, dict):
            raise TelegramError(f'{method}: HTTP {status}, ответ не JSON', code=status,
                                network=status == 200 or status >= 500)
        if not payload.get('ok'):
            parameters = payload.get('parameters') if isinstance(payload.get('parameters'), dict) else {}
            code = payload.get('error_code', status)
            raise TelegramError(f'{method}: {code} {self.redact(payload.get("description", ""))}'.rstrip(),
                                code=code, retry_after=parameters.get('retry_after'),
                                network=isinstance(code, int) and code >= 500)
        return payload.get('result')

    def download(self, file_path, limit, *, timeout=60):
        url = f'{self.base}/file/bot{self._token}/{urllib.parse.quote(file_path, safe="/")}'
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                length = response.headers.get('Content-Length', '')
                if length.isdigit() and int(length) > limit:
                    raise FileTooLarge(limit)
                data = response.read(limit + 1)
        except urllib.error.HTTPError as error:
            error.close()
            raise TelegramError(f'download: HTTP {error.code}', code=error.code) from None
        except (OSError, http.client.HTTPException) as error:
            raise self._network_error('download', error) from None
        if len(data) > limit:
            raise FileTooLarge(limit)
        return data


def score_pair(score, threshold):
    for digits in (4, 6, 8):
        left, right = f'{score:.{digits}f}', f'{threshold:.{digits}f}'
        if left != right:
            break
    return left, right


def format_reply(result, threshold, max_chars=None):
    """max_chars: обрезать список лиц, чтобы ответ уместился в подпись к фото (один ответ на фото)."""
    faces = result.get('faces') or []
    head = [VERDICTS.get(result.get('status'), 'Не удалось проверить фото'), f'Найдено лиц: {len(faces)}']
    footer = ['', 'Учебная модель: оценка — не вероятность и не доказательство личности.']
    if not faces:
        return '\n'.join(head + ['Попробуйте фото, где лицо видно крупнее и ближе к анфас.'] + footer)
    head += [f'Порог модели: {threshold:.4f}', '']
    face_lines = []
    for number, face in enumerate(faces[:MAX_FACE_LINES], 1):
        if face.get('score') is None:
            face_lines.append(f'Лицо {number}: не удалось проверить (жёлтая рамка)')
            continue
        score, limit = score_pair(face['score'], threshold)
        if face.get('positive'):
            face_lines.append(f'Лицо {number}: {score} ≥ {limit} — Соколов (зелёная рамка)')
        else:
            face_lines.append(f'Лицо {number}: {score} < {limit} — не Соколов (красная рамка)')
    shown = len(face_lines)

    def build(count):
        more = [f'…и ещё лиц: {len(faces) - count}'] if count < len(faces) else []
        return '\n'.join(head + face_lines[:count] + more + footer)
    while max_chars and shown and len(build(shown)) > max_chars:
        shown -= 1
    return build(shown)


def annotate(data, faces, *, max_side=1600):
    """JPEG с рамками; координаты — в системе decode_photo (после EXIF-поворота), как в classify."""
    from PIL import Image, ImageDraw, ImageFont
    from sokolov.inference import decode_photo
    image = decode_photo(data)
    scale = min(1.0, max_side/max(image.size))
    if scale < 1:
        image = image.resize((max(1, round(image.width*scale)), max(1, round(image.height*scale))),
                             Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    line = max(2, round(max(image.size)/300))
    try:
        font = ImageFont.load_default(size=max(14, round(max(image.size)/40)))
    except (TypeError, OSError, ImportError):
        font = ImageFont.load_default()
    for number, face in enumerate(faces, 1):
        kind = 'error' if face.get('score') is None else 'positive' if face.get('positive') else 'negative'
        x0, y0, x1, y1 = (value*scale for value in face['bbox'])
        x0, x1 = sorted((min(max(x0, 0), image.width - 1), min(max(x1, 0), image.width - 1)))
        y0, y1 = sorted((min(max(y0, 0), image.height - 1), min(max(y1, 0), image.height - 1)))
        draw.rectangle((x0, y0, x1, y1), outline=COLORS[kind], width=line)
        box = draw.textbbox((0, 0), str(number), font=font)
        width, height = box[2] - box[0] + 2*line, box[3] - box[1] + 2*line
        top = y0 - height if y0 >= height else y0
        draw.rectangle((x0, top, x0 + width, top + height), fill=COLORS[kind])
        draw.text((x0 + line - box[0], top + line - box[1]), str(number), fill='white', font=font)
    stream = BytesIO()
    image.save(stream, format='JPEG', quality=88)
    return stream.getvalue()


class SokolovBot:
    """Обработка одного обновления. Изображения живут только в памяти этого вызова."""

    def __init__(self, api, classifier, *, allowed_chats=frozenset(), max_bytes=MAX_FILE_BYTES,
                 log=log):
        self.api, self.classifier = api, classifier
        self.allowed_chats, self.max_bytes, self.log = frozenset(allowed_chats), max_bytes, log

    def handle(self, update):
        started = time.monotonic()
        message = update.get('message') if isinstance(update, dict) else None
        chat = message.get('chat') if isinstance(message, dict) else None
        chat_id = chat.get('id') if isinstance(chat, dict) else None
        if not isinstance(chat_id, int) or isinstance(chat_id, bool):
            return
        message_id = message.get('message_id')
        if self.allowed_chats and chat_id not in self.allowed_chats:
            self.send_text(chat_id, message_id, 'Этот бот доступен только участникам проекта.')
            return self.record(chat_id, None, 'not_allowed', started)
        source = self.image_source(message)
        if source is None:
            text = message.get('text') if isinstance(message.get('text'), str) else ''
            command = text.split()[0].split('@')[0].lower() if text.strip() else ''
            verdict = 'help' if command in ('/start', '/help') else 'no_image'
            self.send_text(chat_id, message_id, HELP if verdict == 'help' else HINT)
            return self.record(chat_id, None, verdict, started)
        file_id, refusal = source
        if refusal:
            self.send_text(chat_id, message_id, refusal)
            return self.record(chat_id, None, 'rejected', started)
        self.check_photo(chat_id, message_id, file_id, started)

    def image_source(self, message):
        """(file_id, None), (None, текст отказа) или None, если изображения нет."""
        photo, document = message.get('photo'), message.get('document')
        if isinstance(photo, list) and photo:
            sizes = [item for item in photo if isinstance(item, dict) and item.get('file_id')]
            if not sizes:
                return None, NOT_IMAGE
            item = max(sizes, key=lambda p: (p.get('width', 0)*p.get('height', 0), p.get('file_size', 0)))
        elif isinstance(document, dict) and document.get('file_id'):
            if not str(document.get('mime_type', '')).startswith('image/'):
                return None, NOT_IMAGE
            item = document
        else:
            return None
        size = item.get('file_size')
        if isinstance(size, int) and size > self.max_bytes:
            return None, str(FileTooLarge(self.max_bytes))
        return item['file_id'], None

    def check_photo(self, chat_id, message_id, file_id, started):
        try:
            self.api.call('sendChatAction', {'chat_id': chat_id, 'action': 'typing'}, timeout=10)
        except TelegramError:
            pass
        try:
            info = self.api.call('getFile', {'file_id': file_id})
            path = info.get('file_path') if isinstance(info, dict) else None
            if not path:
                raise TelegramError('getFile: нет file_path')
            if isinstance(info.get('file_size'), int) and info['file_size'] > self.max_bytes:
                raise FileTooLarge(self.max_bytes)
            data = self.api.download(path, self.max_bytes)
            inference_started = time.monotonic()
            result = self.classifier.classify(data)
            inference_ms = (time.monotonic() - inference_started)*1000
        except ValueError as error:
            self.send_text(chat_id, message_id, str(error))
            return self.record(chat_id, None, 'rejected', started)
        except TelegramError as error:
            self.log(f'chat={chat_id} {error}')
            self.send_text(chat_id, message_id, 'Не удалось получить фото из Telegram. Отправьте его ещё раз.')
            return self.record(chat_id, None, 'download_error', started)
        except Exception as error:
            self.log(f'chat={chat_id} classify failed: {type(error).__name__}')
            self.send_text(chat_id, message_id, 'Не удалось проверить фото. Попробуйте другое изображение.')
            return self.record(chat_id, None, 'error', started)
        faces = result.get('faces') or []
        text = format_reply(result, self.classifier.threshold)
        text_sent = False
        if faces:
            # Ровно один ответ: фото с рамками и подписью, укороченной до лимита Telegram.
            try:
                image = annotate(data, faces)
                caption = format_reply(result, self.classifier.threshold, max_chars=CAPTION_LIMIT)
                self.api.call('sendPhoto', {'chat_id': chat_id, 'caption': caption,
                                            'reply_parameters': self.reply_to(message_id)},
                              files={'photo': ('faces.jpg', image, 'image/jpeg')}, timeout=60)
                text_sent = True
            except TelegramError as error:
                # После сетевого таймаута фото могло уже дойти: текст не дублируем.
                text_sent = error.network
                self.log(f'chat={chat_id} annotated photo not confirmed: {type(error).__name__}'
                         f'{" (network; text not resent)" if error.network else ""}')
            except Exception as error:
                self.log(f'chat={chat_id} annotated photo not sent: {type(error).__name__}')
        if not text_sent:
            self.send_text(chat_id, message_id, text)
        self.record(chat_id, len(faces), result.get('status'), started, inference_ms)

    @staticmethod
    def reply_to(message_id):
        if isinstance(message_id, int):
            return {'message_id': message_id, 'allow_sending_without_reply': True}
        return None

    def send_text(self, chat_id, message_id, text):
        self.api.call('sendMessage', {'chat_id': chat_id, 'text': text,
                                      'reply_parameters': self.reply_to(message_id)})

    def record(self, chat_id, faces, verdict, started, inference_ms=None):
        line = (f'chat={chat_id} faces={"-" if faces is None else faces} verdict={verdict} '
                f'latency_ms={(time.monotonic() - started)*1000:.0f}')
        if inference_ms is not None:
            line += f' inference_ms={inference_ms:.0f}'
        self.log(line)


def poll(api, handle, *, timeout=30, stop=None, log=log, backoff=(1.0, 60.0)):
    """Long polling. Ошибка одного обновления не останавливает цикл; 401/404 — фатальны."""
    stop = stop or threading.Event()
    offset, delay = None, backoff[0]
    try:
        while not stop.is_set():
            try:
                updates = api.call('getUpdates', {'offset': offset, 'timeout': timeout,
                                                  'allowed_updates': ['message']}, timeout=timeout + 15)
            except TelegramError as error:
                if error.code in FATAL_CODES:
                    raise
                wait = max(delay, error.retry_after if isinstance(error.retry_after, (int, float)) else 0)
                log(f'{error}; повтор через {wait:g} с')
                stop.wait(wait)
                delay = min(delay*2, backoff[1])
                continue
            delay = backoff[0]
            for update in updates if isinstance(updates, list) else ():
                update_id = update.get('update_id') if isinstance(update, dict) else None
                if isinstance(update_id, int) and not isinstance(update_id, bool):
                    offset = max(offset or 0, update_id + 1)
                try:
                    handle(update)
                except Exception as error:
                    log(f'update={update_id} не обработан: {type(error).__name__}: {api.redact(error)}')
    finally:
        if offset is not None:
            try:
                api.call('getUpdates', {'offset': offset, 'timeout': 0, 'limit': 1}, timeout=5)
            except Exception:
                pass
    return offset


def _interrupt(*_):
    raise KeyboardInterrupt


def main(argv=None):
    parser = argparse.ArgumentParser(description='Telegram-бот учебного классификатора Соколова')
    parser.add_argument('--run-dir', type=Path, default=Path('outputs/runs/now-v3-eq'))
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    parser.add_argument('--detector-root', type=Path)
    parser.add_argument('--token-file', type=Path,
                        help='файл с токеном (по умолчанию TELEGRAM_BOT_TOKEN, затем work/telegram_token)')
    parser.add_argument('--api-base', default=DEFAULT_API_BASE, help='для тестов с поддельным API')
    parser.add_argument('--poll-timeout', type=int, default=30)
    parser.add_argument('--max-file-mb', type=float, default=MAX_FILE_BYTES/1024/1024)
    args = parser.parse_args(argv)
    try:
        allowed = parse_allowed(os.environ.get('TELEGRAM_ALLOWED_CHATS', ''))
        token, source = load_token(args.token_file)
        api = TelegramAPI(token, args.api_base)
    except ValueError as error:
        print(f'Ошибка: {error}', file=sys.stderr, flush=True)
        return 2
    run_dir = args.run_dir.resolve()
    detector_root = args.detector_root.resolve() if args.detector_root else None
    os.chdir(ROOT)
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'work/matplotlib'))
    os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '0')
    if str(ROOT/'src') not in sys.path:
        sys.path.insert(0, str(ROOT/'src'))
    signal.signal(signal.SIGTERM, _interrupt)
    try:
        try:
            me = api.call('getMe', timeout=15)
            log(f'Telegram: @{me.get("username")} (токен: {source})')
        except TelegramError as error:
            if error.code in FATAL_CODES:
                raise
            log(f'Предупреждение: {error}; продолжаю, повторы будут в цикле опроса')
        log('Загружаю модель и детектор…')
        from sokolov.inference import PhotoClassifier
        classifier = PhotoClassifier(run_dir, device=args.device, detector_root=detector_root)
        bot = SokolovBot(api, classifier, allowed_chats=allowed, max_bytes=int(args.max_file_mb*1024*1024))
        log(f'Бот запущен: {classifier.architecture} · {classifier.device} · порог {classifier.threshold:.4f} · '
            f'доступ: {"все" if not allowed else f"{len(allowed)} чат(ов)"}')
        poll(api, bot.handle, timeout=args.poll_timeout)
    except KeyboardInterrupt:
        log('Бот остановлен.')
    except TelegramError as error:
        log(f'Ошибка Telegram: {error}' + (' — проверьте токен' if error.code in FATAL_CODES else ''))
        return 2
    except Exception as error:
        log(f'Ошибка: {type(error).__name__}: {api.redact(error)}')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
