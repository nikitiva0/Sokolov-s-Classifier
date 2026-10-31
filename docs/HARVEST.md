# Сбор лиц из публичных источников (`tools/harvest_faces.py`)

Инструмент собирает кропы лиц для `data/raw/web_positive_v2/` (источники с Евгением Соколовым) и
`data/raw/web_negative_v2/` (другие люди в похожих условиях — трудные негативы). Формат строго по
HARVEST CONTRACT: один кроп = одно лицо, `manifest.csv` с фиксированными колонками,
`data/sources/<dataset>.json` с метаданными источников. Изображения в Git не попадают
(`data/raw/*` игнорируется); в Git идут только код, пороги и `data/sources/*.json` без изображений.

## Окружение

```bash
cd "/Users/skara/HSE/Проектная школа/Sokolov-s-Classifier" && source .venv/bin/activate
python tools/harvest_faces.py --help
```

- Модели: пакет InsightFace `buffalo_l` в `checkpoints/insightface/models/buffalo_l/`
  (SCRFD `det_10g.onnx` — тот же детектор и порог, что в `sokolov.data.prepare`; ArcFace
  `w600k_r50.onnx` — эмбеддинги). Выравнивание — `sokolov.data.prepare.align_face` (224×224),
  затем INTER_AREA до 112×112 px.
- Внешние программы и пакеты: `yt-dlp` и `imageio-ffmpeg` (ffmpeg 7.1) — уже стоят в `.venv`,
  но **не перечислены в `pyproject.toml`**; `dig` (системный, macOS) — резервный DNS;
  `node` — необязателен, передаётся yt-dlp как JS-runtime для YouTube. Новых зависимостей нет.
- Все сетевые команды запускайте с отключённой песочницей (прокси песочницы ≈ 50 KB/s).

## Быстрый старт

```bash
# 1. Эталон: 1000 позитивов v1 → data/reference/ (локально, ~5 мин на CPU)
python tools/harvest_faces.py reference

# 2. Пороги (уже откалиброваны, tools/harvest/thresholds.json); пересчёт:
python tools/harvest_faces.py calibrate --negatives <manifest.csv лиц заведомо других людей> --note "кто это"

# 3. Видео с Соколовым одним в кадре (веб-камера / сцена)
python tools/harvest_faces.py video "https://rutube.ru/video/<id>/" --dataset web_positive_v2 \
    --source-id rutube_sokolov_talk1 --domain talk_stage --mode solo --count 60 \
    --verification "сверено с preview.jpg: борода, очки, голос; страница ФКН"

# 4. Видео, где Соколов среди других людей (панель, семинар)
python tools/harvest_faces.py video URL --dataset web_positive_v2 --source-id panel1 \
    --domain interview --mode multi --every 30

# 5. Другой лектор (трудный негатив); --person — самому крупному лицу кадра
python tools/harvest_faces.py video URL --dataset web_negative_v2 --source-id lect_ivanov_ml1 \
    --domain lecture_room --mode negative --person ivanov_ii --count 40 --max-faces 1

# 6. Фото: по URL или из каталога уже скачанных файлов
python tools/harvest_faces.py images URL1 URL2 --dataset web_positive_v2 --source-id hse_page \
    --mode solo --title "Страница сотрудника" --date 2025-09-01
python tools/harvest_faces.py images --dir data/raw/_scratch/photos_x --source-url https://... \
    --dataset web_negative_v2 --source-id event_x --mode negative

# 7. Аудит
python tools/harvest_faces.py sheet --dataset web_positive_v2 --source panel1 --sort sim
python tools/harvest_faces.py sheet --dataset web_positive_v2 --status ambiguous
python tools/harvest_faces.py set-status --dataset web_positive_v2 --ids wp2_panel1_0007 wp2_panel1_0011 \
    --status excluded --note "не Соколов: студент в очках"
python tools/harvest_faces.py set-status --dataset web_positive_v2 --ids wp2_panel1_0003 \
    --status accepted --label 1 --note "Соколов, профиль; сверено с v1"
python tools/harvest_faces.py dedup-v1 --dataset web_positive_v2 [--dry-run]
python tools/harvest_faces.py stats --dataset web_positive_v2
```

Общие параметры сбора: `--max-faces N` (N крупнейших лиц кадра), `--min-det 0.6`, `--min-face 28`
(px, max(w, h) bbox), `--jobs 3` (параллельных загрузок), `--start/--end` (s),
`--max-height 720`, `--max-kbps 2500`, `--fetch auto|hls|seek|download`, `--exact`, `--max-seconds`,
`--max-failures 8`, `--stall 180` (s), `--max-resolves 3`, `--ytdlp-args "--cookies FILE"`.
Глобальный `--root DIR` перенаправляет все данные (manifest, кропы, sources) в другой корень —
для тестов, чтобы не трогать общие файлы.

## Что делает `video` / `images`

1. `yt-dlp -J` в подпроцессе (timeout 120 s; `--socket-timeout 20`, повторы). Для YouTube
   перебираются `player_client` = default, web_safari, tv, ios, mweb, android_vr.
2. Формат: видео ≤ `--max-height` (по умолчанию 720p), не AV1, без DRM, протокол HLS или
   http(s); форматы с известным битрейтом > `--max-kbps` отбрасываются, если есть другие
   (ключевой кадр 720p при ~5 Mbit/s не помещается в пробы и тянет весь сегмент ≈ 2.3 MiB);
   при равной высоте — HLS, затем меньший битрейт.
3. Стратегия (`--fetch auto`):
   - `download` — если оценка размера файла ≤ `--download-max-mib` (48 MiB): файл целиком в
     `data/raw/_scratch/downloads/`, кадры локально, файл удаляется в `finally`. Сторож: нет роста
     файла 120 s → перезапуск (до 3 раз);
   - `hls` — плейлист читается один раз; кадр = начало сегмента, содержащего t (там ключевой кадр);
     скачивается только начало сегмента Range-запросом 160 KiB → 512 KiB → весь сегмент, пока
     декодер не отработает **без ошибок** (обрезанный кадр ffmpeg выдаёт молча — с битым низом —
     и сообщает об этом только в логе; такие кадры отбрасываются);
   - `seek` — `ffmpeg -ss t -noaccurate_seek -i URL` (ключевой кадр ≤ t; `--exact` — точный кадр,
     дороже), `-rw_timeout 15 s`, timeout процесса 60 s, 3 попытки.
4. Каждый HTTP-запрос: timeout 8 s на connect/read, 4 попытки с экспоненциальной паузой, общий
   дедлайн 90 s; HTTP 429 — пауза по Retry-After (≤ 60 s). Сбой узла CDN или истёкшая ссылка →
   новый resolve (до `--max-resolves`, обычно выдаётся другой узел).
5. Останов без «слепого» ожидания: `--max-failures` неудач подряд, ни одного готового кадра за
   `--stall` s, общий бюджет `--max-seconds`. Каждая строка лога — один кадр с временем.
6. Лица: SCRFD (порог конфига 0.5), затем фильтр det ≥ 0.6 и face_px ≥ 28 px; выравнивание,
   ArcFace, `sokolov_similarity` = cos к центроиду v1. Лицо с cos ≥ 0.97 к уже сохранённому
   (не excluded) лицу того же источника пропускается как почти дубликат.
7. Метка по режиму и порогам (ниже), кроп по контракту (квадрат 2 × max(w, h), сдвиг внутрь кадра,
   уменьшение только если кадр меньше; JPEG q95; > 512 px — уменьшение до 512), строка manifest.
8. Итог: число кадров/ошибок, счётчики лиц, байты и секунды, контактный лист
   `data/raw/<dataset>/sheets/<source_id>.jpg`, обновление `data/sources/<dataset>.json`.

Учёт байтов: медиа-байты (HTTP-тела, `Statistics: N bytes read` ffmpeg, размер скачанного файла).
Запросы yt-dlp к страницам (обычно сотни KiB на resolve) не учитываются.

## Идемпотентность и возобновление

- `data/raw/<dataset>/state/<source_id>.json`: статус каждого запрошенного момента / URL;
  повторный запуск с теми же параметрами обрабатывает только не-`ok` кадры. Прерывание (Ctrl-C,
  сбой сети) безопасно: состояние пишется после каждого кадра.
- `manifest.csv` дополняется под `flock`; строка с тем же `image_id` или тем же естественным ключом
  (source_id, timestamp_seconds, source_url, det_score, face_px) не добавляется второй раз.
  Нумерация `wp2_<source>_<nnnn>` продолжается с максимального номера.
- Один источник — один процесс (`state/<source_id>.lock`); разные источники можно собирать
  параллельно.
- `data/raw/<dataset>/embeddings/<source_id>.npz` — эмбеддинги, pHash, bbox, landmarks (для
  почти-дубликатов и `dedup-v1`).

## Режимы и пороги

`sokolov_similarity` — косинус эмбеддинга ArcFace (w600k_r50) к L2-нормированному центроиду
994 лиц v1. Пороги лежат в `tools/harvest/thresholds.json` вместе с измеренным обоснованием
(`basis`) и правилом (`rule`).

| Режим | label 1 | label 0 | ambiguous |
|---|---|---|---|
| `solo` (Соколов один в источнике) | sim ≥ T_solo_min | — | sim < T_solo_min (метка 1 предварительная) |
| `multi` (Соколов среди других) | sim ≥ T_pos | sim ≤ T_neg, person=unknown, group=source | между порогами |
| `negative` (Соколова нет) | — | sim < T_flag | sim ≥ T_flag: «возможно Соколов» |

Группы: Соколов → `group_id = source_id`; известный человек → `person:<slug>` (одинаково во всех
видео); неизвестные → `source_id`. Только `accepted` идут дальше; `ambiguous` и `excluded`
остаются в manifest для аудита.

Эталон (`reference`, 2026-10-31): лицо найдено на 994 из 1000 кадров v1 (99.4 %: лекция 01 —
249/250, 03 — 245/250, 04 и 11 — 250/250), несколько лиц — ни на одном. Сходство с центроидом
остальных трёх лекций (leave-one-lecture-out, LOLO): min 0.064, p1 0.305, p5 0.482, медиана 0.715
(по лекциям медианы 0.763 / 0.702 / 0.692 / 0.713). Размер лица v1: медиана 57.7 px, минимум 30.5 px.

Impostor — 58 лиц ≈ 15 заведомо других людей (лекторы Rutube Teach-in/МГУ, MSU_AI, СПбГУ и лица
со слайдов; спикеры Wikimedia Commons; `t1.jpg` из пакета insightface), все проверены глазами:
min −0.088, медиана −0.002, p95 0.088, max 0.123 (самый похожий — бородатый лектор СПбГУ).

| Порог | Значение | Правило | Доля genuine LOLO ниже | Доля impostor ≥ |
|---|---|---|---|---|
| T_neg = T_flag | 0.17 | max(impostor) + 0.05 | 0.3 % | 0 % |
| T_solo_min | 0.33 | (T_neg + T_pos) / 2 | 1.3 % | 0 % |
| T_pos | 0.48 | max(T_neg + 0.10, genuine LOLO p5) | 4.9 % | 0 % |
| near_duplicate | 0.97 | из задания | — | — |
| v1_cos / v1_phash | 0.92 / 6 бит | из задания, см. dedup-v1 | — | — |

T_pos выбран консервативно: ложная метка 1 вредит больше, чем лишняя строка на аудите.
Неопределённость: genuine — одна веб-камера, impostor — мало личностей и почти нет «двойников»
(борода + очки + возраст); хвост чужих лиц оценён грубо. Пересчёт: `calibrate` с новыми
manifest проверенных чужих лиц.

## Аудит

- `sheet`: до 8×8 лиц на изображение, рамка — статус/метка (зелёная = 1 accepted, синяя = 0
  accepted, оранжевая = ambiguous, серая = excluded), подпись — номер, метка, статус, время и sim.
  `--sort sim` удобно для поиска чужих лиц в позитивном источнике.
- Личность всегда подтверждается глазами по `data/raw/sokolov_positive_v1/preview.jpg`: имя
  «Евгений Соколов» носят многие люди. Текст проверки — в `--verification` (поле
  `identity_verification` источника).
- `set-status` меняет статус (и при `--label` — метку, person, group_id); счётчики в
  `data/sources` пересчитываются.

### dedup-v1

Помечает `accepted` → `ambiguous`, если лицо похоже на позитив v1: pHash выровненного кропа
≤ 6 бит **или** cos эмбеддингов ≥ 0.92. Измерено на самих v1 (994 лица, ближайший сосед):

| Признак (ближайший сосед в v1) | та же лекция | другая лекция |
|---|---|---|
| pHash выровненного кропа ≤ 6 бит | 53.3 % лиц | 8.2 % лиц |
| cos эмбеддингов ≥ 0.92 | 0 % (max 0.908) | 0 % (max 0.884) |
| pHash, медиана расстояния | 6 бит | 12 бит |

Выровненные лица структурно похожи, поэтому pHash ≤ 6 даёт ≈ 8 % ложных флагов для другой записи,
а cos ≥ 0.92 срабатывает только на практически тот же кадр (кадры v1 идут с шагом ≈ 17 s).

Поэтому отдельное лицо с флагом — не доказательство повторной записи; решающим считается доля
флагов по источнику (предупреждение при ≥ 25 % и ≥ 10 лицах), после чего источник проверяется
целиком. Ложные флаги возвращаются `set-status --status accepted`.

## Измеренная производительность

31.10.2026, Apple M4, канал ≈ 750 KB/s делился с параллельными загрузками других агентов —
цифры ориентировочные. «Фаза кадров» — от первой загрузки кадра до последнего лица, без resolve
(1.4–1.8 s для Rutube) и загрузки моделей с плейлистом (1.4–2.2 s).

| Источник, стратегия | Кадров ок | Фаза кадров | KiB/кадр | KiB/s | s/кадр |
|---|---|---|---|---|---|
| Rutube, HLS 720p 433 kbit/s (итоговый код) | 30/30 | 14.0 s | 252 | 541 | 0.47 |
| Rutube, HLS 720p (MSU_AI) | 16/16 | 6.1 s | 220 | 580 | 0.38 |
| Rutube, HLS 720p 4951 kbit/s, недоступный узел CDN (до `--max-kbps`) | 11/16 | 335 s | 2367 | 78 | 30.5 |
| Wikimedia, `download` 480p VP9, файл 19 264 KiB | 8/8 | 1.8 s локально; всего 28.0 s | — | 688 всего | 0.22 локально |
| Wikimedia, `seek` 480p (принудительно) | 2/8 (6 × HTTP 429) | 35.3 s | 153 | 9 | 17.7 |
| Фото с диска 1280×886 px (CPU) | 2/2 | 0.7 s | 0 | — | 0.35 |
| YouTube | отказ за 14.4 s (6 клиентов) | — | — | — | — |

Обрезанный HLS-сегмент: на одном и том же сегменте 720p/433 kbit/s первые 24–96 KiB дают кадр с
битым низом (ошибки декодера), 128 KiB — почти полный (битые последние 32–48 строк из 720); поэтому
пробы 160 KiB → 512 KiB → весь сегмент с проверкой лога декодера. Недоступный узел CDN
распознаётся за ≈ 4 s на попытку (connect timeout).

## Известные ограничения

- YouTube с этого IP требует вход («Sign in to confirm you're not a bot») для всех клиентов
  yt-dlp; cookies нет. Инструмент выдаёт понятную ошибку; используйте VK Video, Rutube, прямые
  mp4 или `--ytdlp-args "--cookies FILE"`.
- Системный резолвер macOS здесь не разрешает отдельные имена (www.youtube.com), хотя DNS-сервер
  отвечает; yt-dlp и HTTP-запросы инструмента используют резерв через `dig`. ffmpeg (режим
  `seek`) резерва не имеет.
- Часть узлов CDN Rutube отсюда недоступна (TCP connect timeout); помогает новый resolve, но
  балансировщик иногда снова выдаёт тот же узел — тогда кадры идут медленно или с ошибками.
- upload.wikimedia.org отвечает 429 на частые Range-запросы: для Commons используйте `download`
  (выбирается автоматически для коротких файлов) и `--jobs 1`.
- Эталон — 4 лекции одной веб-камеры: на сцене, в интервью и на фото сходство Соколова ниже, чем
  в v1. `ambiguous` в solo-источниках обязательно смотреть глазами.
