"""Сеть: жёсткие таймауты, повторы, учёт байтов. yt-dlp — только в подпроцессе с timeout.

Кадры берутся без скачивания целого видео: HLS — частичным Range-запросом начала сегмента
(сегмент начинается с ключевого кадра), прогрессивный файл — ffmpeg -ss до -i (быстрый seek).
"""
from __future__ import annotations
import http.client
import json
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urljoin, urlparse

UA = ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 '
      '(KHTML, like Gecko) Chrome/130.0 Safari/537.36')
RUNNER = Path(__file__).with_name('ytdlp_run.py')
YOUTUBE_CLIENTS = ('web_safari', 'tv', 'ios', 'mweb', 'android_vr')
VIDEO_PROTOCOLS = ('https', 'http', 'm3u8_native', 'm3u8')


class FetchError(RuntimeError):
    def __init__(self, message, code=None):
        super().__init__(message)
        self.code = code


class ByteCounter:
    def __init__(self):
        self.lock, self.value = threading.Lock(), 0

    def add(self, n):
        with self.lock:
            self.value += int(n)


BYTES = ByteCounter()

# --- DNS: системный резолвер этой машины не разрешает отдельные имена (www.youtube.com),
# хотя DNS-сервер отвечает. Резерв — dig, только при gaierror.
_system_getaddrinfo = socket.getaddrinfo
_dns_cache = {}


def _getaddrinfo(host, *args, **kwargs):
    try:
        return _system_getaddrinfo(host, *args, **kwargs)
    except socket.gaierror as error:
        if not isinstance(host, str) or not shutil.which('dig'):
            raise
        if host not in _dns_cache:
            try:
                out = subprocess.run(['dig', '+short', '+time=3', '+tries=2', host, 'A'],
                                     capture_output=True, text=True, timeout=15).stdout
            except (OSError, subprocess.TimeoutExpired):
                raise error from None
            ips = [x for x in out.split() if re.fullmatch(r'\d+(\.\d+){3}', x)]
            if not ips:
                raise
            _dns_cache[host] = ips[0]
        return _system_getaddrinfo(_dns_cache[host], *args, **kwargs)


def install_dns_fallback():
    socket.getaddrinfo = _getaddrinfo


def ffmpeg_exe():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


CONNECT_TIMEOUT = 4.0  # s; узлы CDN отсюда часто не отвечают на SYN — быстрее перейти к повтору


class _Connection(http.client.HTTPSConnection):
    """Короткий таймаут на TCP+TLS, обычный — на чтение."""

    def connect(self):
        read_timeout, self.timeout = self.timeout, min(self.timeout or CONNECT_TIMEOUT, CONNECT_TIMEOUT)
        try:
            super().connect()
        finally:
            self.timeout = read_timeout
        self.sock.settimeout(read_timeout)


class _Handler(urllib.request.HTTPSHandler):
    def https_open(self, request):
        return self.do_open(_Connection, request, context=self._context)


_OPENER = urllib.request.build_opener(_Handler())


def http_get(url, headers=None, byte_range=None, timeout=10, attempts=6, deadline=90, max_bytes=64 << 20):
    """GET: connect ≤ 4 s, чтение ≤ timeout s, повторы с паузой ≤ 4 s, общий дедлайн (s)."""
    request_headers = {'User-Agent': UA, **(headers or {})}
    limit = max_bytes
    if byte_range:
        request_headers['Range'] = f'bytes={byte_range[0]}-{byte_range[1]}'
        limit = byte_range[1] - byte_range[0] + 1
    start, last, pause = time.monotonic(), None, 0
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers=request_headers)
            with _OPENER.open(request, timeout=timeout) as response:
                chunks, size = [], 0
                while size < limit:
                    chunk = response.read(min(65536, limit - size))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    size += len(chunk)
                    BYTES.add(len(chunk))
                    if time.monotonic() - start > deadline:
                        raise FetchError(f'дедлайн {deadline} s при чтении')
                if size >= max_bytes and not byte_range:
                    raise FetchError(f'ответ больше {max_bytes >> 20} MiB', 413)
                return b''.join(chunks)
        except urllib.error.HTTPError as error:
            last = error
            if error.code in (400, 401, 403, 404, 410, 451):
                raise FetchError(f'HTTP {error.code} {urlparse(url).netloc}', error.code) from None
            if error.code == 429:  # ограничение частоты: ждём Retry-After (≤ 60 s)
                retry = error.headers.get('Retry-After', '')
                pause = min(60, int(retry) if retry.isdigit() else 10 * 2 ** attempt)
        except (OSError, http.client.HTTPException, FetchError) as error:
            if getattr(error, 'code', None) == 413:  # повтор не поможет
                raise
            last = error
        pause, wait_s = 0, max(pause, min(4, .5 * 2 ** attempt))
        if time.monotonic() - start + wait_s > deadline:
            break
        time.sleep(wait_s)
    raise FetchError(f'{urlparse(url).netloc}: {type(last).__name__}: {last} (попыток: {attempt + 1})')


# --- yt-dlp ---------------------------------------------------------------------------

def is_youtube(url):
    host = urlparse(url).netloc.lower()
    return host.endswith(('youtube.com', 'youtu.be', 'youtube-nocookie.com'))


def ytdlp_command(args):
    command = [sys.executable, str(RUNNER), '--socket-timeout', '20', '--retries', '3',
               '--extractor-retries', '2', '--no-playlist']
    if shutil.which('node'):
        command += ['--js-runtimes', 'node']
    return command + list(args)


def resolve(url, extra_args=(), timeout=120):
    """yt-dlp -J; для YouTube перебор player_client. Возвращает (info, использованные аргументы)."""
    variants = [[]]
    if is_youtube(url):
        variants += [['--extractor-args', f'youtube:player_client={c}'] for c in YOUTUBE_CLIENTS]
    errors = []
    for variant in variants:
        args = [*variant, *extra_args]
        started = time.monotonic()
        try:
            result = subprocess.run(ytdlp_command(['-J', *args, url]), capture_output=True,
                                    text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            errors.append(f'{variant or "default"}: таймаут {timeout} s')
            continue
        if result.returncode == 0 and result.stdout.strip():
            info = json.loads(result.stdout)
            info['_resolve_seconds'] = round(time.monotonic() - started, 2)
            return info, args
        lines = [x for x in result.stderr.splitlines() if 'ERROR' in x] or result.stderr.splitlines()[-1:]
        message = re.sub(r' Use --cookies.*', '', lines[-1] if lines else f'код {result.returncode}')[:240]
        errors.append(f'{" ".join(variant) or "default"}: {message}')
        if 'Unsupported URL' in message or 'HTTP Error 404' in message:
            break
    hint = ''
    if any('not a bot' in e or 'Sign in' in e for e in errors):
        hint = ('\nYouTube требует вход (проверка на бота) для этого IP; cookies нет. Возьмите тот же '
                'материал с VK Video/Rutube/прямого mp4 или передайте --ytdlp-args "--cookies FILE".')
    raise FetchError('yt-dlp не смог получить видео:\n  ' + '\n  '.join(errors) + hint)


def pick_format(info, max_height=720, max_kbps=2500):
    """Видео ≤ max_height без AV1/DRM; HLS или прогрессивный http(s). Выше — лучше; при равной
    высоте HLS (дешёвый частичный доступ), затем меньший битрейт. Форматы с известным битрейтом
    > max_kbps отбрасываются, если есть другие: ключевой кадр 720p при ~5 Mbit/s > 512 KiB."""
    candidates = []
    for f in info.get('formats') or [info]:
        vcodec = f.get('vcodec')
        if vcodec == 'none' or (vcodec is None and f.get('ext') not in ('mp4', 'webm', 'mov', 'mkv', 'ts', 'm3u8')
                                and f.get('protocol') not in ('m3u8', 'm3u8_native')):
            continue
        if (vcodec or '').startswith(('av01', 'av1')) or f.get('has_drm') or not f.get('url'):
            continue
        if f.get('protocol', 'https') not in VIDEO_PROTOCOLS or (f.get('height') or 0) > max_height:
            continue
        candidates.append(f)
    if not candidates:
        raise FetchError(f'Нет подходящего видеоформата ≤{max_height}p (HLS/http, не AV1)')
    cheap = [f for f in candidates if not f.get('tbr') or f['tbr'] <= max_kbps]
    candidates = cheap or [min(candidates, key=lambda f: f['tbr'])]
    return max(candidates, key=lambda f: (f.get('height') or 0, f.get('protocol', '').startswith('m3u8'),
                                          -(f.get('tbr') or 0)))


def estimate_bytes(fmt, duration):
    if fmt.get('filesize') or fmt.get('filesize_approx'):
        return int(fmt.get('filesize') or fmt.get('filesize_approx'))
    return int((fmt.get('tbr') or 0) * 1000 / 8 * (duration or 0)) or None


def download(url, fmt_id, destination, extra_args=(), stall=120, attempts=3, timeout=3600):
    """Скачать один формат целиком (только короткие источники). Обрыв, если файл не растёт stall s."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    template = str(destination.with_suffix('')) + '.%(ext)s'
    pattern = destination.stem + '*'
    for attempt in range(attempts):
        process = subprocess.Popen(ytdlp_command(['-f', fmt_id, '--fragment-retries', '10', '--newline',
                                                  '--no-progress', '-o', template, *extra_args, url]),
                                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        started = last_change = time.monotonic()
        last_size = -1
        while process.poll() is None:
            time.sleep(3)
            size = sum(p.stat().st_size for p in destination.parent.glob(pattern) if p.is_file())
            if size != last_size:
                last_size, last_change = size, time.monotonic()
            if time.monotonic() - last_change > stall or time.monotonic() - started > timeout:
                process.kill()
                process.wait(10)
                print(f'  загрузка: нет роста {stall} s или общий таймаут — перезапуск '
                      f'({attempt + 1}/{attempts})', flush=True)
                break
        if process.returncode == 0:
            files = [p for p in destination.parent.glob(pattern)
                     if p.is_file() and not p.name.endswith(('.part', '.ytdl'))]
            if files:
                BYTES.add(files[0].stat().st_size)
                return files[0]
        error = (process.stderr.read() if process.stderr else '')[-300:]
        print(f'  загрузка не удалась: {error.strip()}', flush=True)
    raise FetchError(f'Не удалось скачать формат {fmt_id} за {attempts} попытки')


# --- кадры ----------------------------------------------------------------------------

# Обрезанный кадр ffmpeg всё равно выдаёт (низ серый/битый), сообщая об этом только в логе декодера.
DECODE_ERRORS = re.compile(r'^\[(?:h264|hevc|vp8|vp9|mpeg4|mpeg2video|av1)[^\]]*\].*'
                           r'(?:error|concealing|corrupt|invalid|missing|illegal)|corrupt decoded frame', re.I | re.M)


def decode_first_frame(data, timeout=30):
    """Первый видеокадр из байтов контейнера (TS/fMP4) локальным ffmpeg → PIL RGB или None,
    если кадра нет либо декодер сообщил об ошибке (кадр неполный)."""
    from io import BytesIO
    from PIL import Image
    try:
        result = subprocess.run([ffmpeg_exe(), '-nostdin', '-v', 'error', '-i', 'pipe:0', '-map', '0:v:0',
                                 '-frames:v', '1', '-f', 'image2pipe', '-c:v', 'png', 'pipe:1'],
                                input=data, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    if not result.stdout or DECODE_ERRORS.search(result.stderr.decode('utf-8', 'replace')):
        return None
    with Image.open(BytesIO(result.stdout)) as image:
        return image.convert('RGB')


def ffmpeg_frame(source, t, headers=None, exact=False, timeout=60, attempts=3):
    """Кадр в момент t (s): -ss до -i; без exact — ближайший ключевой кадр ≤ t (дешевле).
    Возвращает (PIL RGB, фактическое время s, прочитано байт)."""
    from io import BytesIO
    from PIL import Image
    remote = '://' in str(source)
    command = [ffmpeg_exe(), '-nostdin', '-hide_banner', '-v', 'verbose']
    if remote:
        command += ['-rw_timeout', '15000000']
        if headers:
            command += ['-headers', ''.join(f'{k}: {v}\r\n' for k, v in headers.items())]
    command += ['-copyts', '-ss', f'{t:.3f}', *([] if exact else ['-noaccurate_seek']), '-i', str(source),
                '-map', '0:v:0', '-frames:v', '1', '-vf', 'showinfo', '-f', 'image2pipe', '-c:v', 'png', 'pipe:1']
    last = ''
    for attempt in range(attempts):
        try:
            result = subprocess.run(command, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            last = f'таймаут {timeout} s'
        else:
            log = result.stderr.decode('utf-8', 'replace')
            read = sum(int(x) for x in re.findall(r'Statistics: (\d+) bytes read', log)) if remote else 0
            BYTES.add(read)
            if result.returncode == 0 and result.stdout and not DECODE_ERRORS.search(log):
                times = re.findall(r'pts_time:\s*([-\d.]+)', log)
                with Image.open(BytesIO(result.stdout)) as image:
                    return image.convert('RGB'), float(times[0]) if times else t, read
            last = ' | '.join(log.strip().splitlines()[-2:])[-300:]
            if re.search(r'HTTP error 40[34]|Server returned 40[34]', log):
                raise FetchError(f'ffmpeg: {last}', 403)
            if '429' in last:  # ограничение частоты (например, upload.wikimedia.org): долгая пауза
                time.sleep(20 * (attempt + 1))
                continue
        if attempt + 1 < attempts:
            time.sleep(1 + 2 * attempt)
    raise FetchError(f'ffmpeg t={t:.1f} s: {last}')


class HLS:
    """Медиаплейлист HLS: кадр = начало сегмента, содержащего t (ключевой кадр)."""

    def __init__(self, url, headers=None, max_height=720):
        self.headers = headers or {}
        text = http_get(url, self.headers, attempts=4).decode('utf-8', 'replace')
        if '#EXT-X-STREAM-INF' in text:
            url = self._variant(url, text, max_height)
            text = http_get(url, self.headers, attempts=4).decode('utf-8', 'replace')
        self.url, self.segments, self.init = url, [], None
        self._init_bytes, self._lock = None, threading.Lock()
        position, duration, byterange, offset = 0.0, None, None, 0
        for line in text.splitlines():
            line = line.strip()
            if line.startswith('#EXT-X-KEY') and 'METHOD=NONE' not in line:
                raise FetchError('HLS зашифрован (EXT-X-KEY) — используйте --fetch seek')
            if line.startswith('#EXT-X-MAP:'):
                uri = re.search(r'URI="([^"]+)"', line).group(1)
                br = re.search(r'BYTERANGE="(\d+)@(\d+)"', line)
                self.init = (urljoin(url, uri), (int(br.group(2)), int(br.group(2)) + int(br.group(1)) - 1) if br else None)
            elif line.startswith('#EXTINF:'):
                duration = float(line[8:].split(',')[0])
            elif line.startswith('#EXT-X-BYTERANGE:'):
                length, _, start = line[17:].partition('@')
                start = int(start) if start else offset
                byterange, offset = (start, start + int(length) - 1), start + int(length)
            elif line and not line.startswith('#') and duration is not None:
                self.segments.append((position, duration, urljoin(url, line), byterange))
                position, duration, byterange = position + duration, None, None
        if not self.segments:
            raise FetchError('Пустой HLS-плейлист')
        self.duration = position

    @staticmethod
    def _variant(url, text, max_height):
        options, attributes = [], None
        for line in text.splitlines():
            if line.startswith('#EXT-X-STREAM-INF'):
                attributes = line
            elif attributes and line.strip() and not line.startswith('#'):
                height = re.search(r'RESOLUTION=\d+x(\d+)', attributes)
                options.append((int(height.group(1)) if height else 0, urljoin(url, line.strip())))
                attributes = None
        fitting = [o for o in options if o[0] <= max_height] or sorted(options)[:1]
        return max(fitting)[1]

    def frame(self, t, probes=(160 << 10, 512 << 10)):
        """(PIL RGB, время начала сегмента s). Начало сегмента Range-запросом растущего размера,
        затем сегмент целиком; кадр принимается только без ошибок декодера."""
        index = next((i for i, s in enumerate(self.segments) if s[0] + s[1] > t), len(self.segments) - 1)
        start, duration, uri, byterange = self.segments[index]
        prefix = b''
        if self.init:
            with self._lock:
                if self._init_bytes is None:
                    self._init_bytes = http_get(self.init[0], self.headers, self.init[1])
            prefix = self._init_bytes
        first, last = byterange or (0, None)
        for size in (*probes, None):
            whole = size is None or (last is not None and first + size - 1 >= last)
            data = http_get(uri, self.headers, byterange if whole else (first, first + size - 1))
            image = decode_first_frame(prefix + data)
            if image is not None:
                return image, start
            if whole or len(data) < size:  # уже весь сегмент
                break
        raise FetchError(f'Сегмент t={start:.1f} s не декодируется без ошибок')
