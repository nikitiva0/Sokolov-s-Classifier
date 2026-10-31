# Как нейросеть представляет Евгения Соколова

Учебный бинарный классификатор: **Соколов / не Соколов**. Обе модели обучаются только на выделенных и выровненных лицах: **ResNet18** (ImageNet, 224×224) и **ArcFace iResNet50** (MS1MV3, 112×112).

Реализованы загрузка FairFace, объединение с позитивами, аудит и разбиение, SCRFD и выравнивание, обучение с продолжением, оценка, сравнительный Markdown/HTML-отчёт и отдельная публикация отчёта в GitHub. Добавлены генерация из шума с градиентной оптимизацией пикселей и сравнение пяти начальных шумов на лучшей сохранённой модели.

**Полный локальный запуск на Apple MPS выполнен.** После исправления масштаба SCRFD сохранены 4944 негативных и 998 позитивных лиц; test-accuracy ResNet18 — 100%, ArcFace — 99,78% на выбранном случайном разбиении. [Проверки и ограничения](docs/VALIDATION.md). Это не оценка качества на новой лекции; веса и полный отчёт хранятся локально в `outputs/runs/local-mps-v2/`.

## Запуск на Mac

[Локальный ноутбук](notebooks/local_train_mps.ipynb) и [инструкция для Apple MPS](docs/LOCAL_MPS.md).
Запуск: `.venv/bin/python tools/run_local_notebook.py`. Он сохраняет выполненный ноутбук,
продолжает обучение с checkpoint и использует строго `mps` без CPU fallback.
Панель фактического прогресса: `.venv/bin/python tools/monitor_training.py --run-dir outputs/runs/local-mps-v2`,
затем открыть http://127.0.0.1:8765.

## Запуск на Kaggle

Откройте [инструкцию](docs/KAGGLE.md) и импортируйте [ноутбук](notebooks/kaggle_train.ipynb).

Чтобы передать текущий код в Kaggle без предварительной публикации в GitHub:

```bash
python3 tools/package_kaggle.py
```

Полученный `work/project_bundle.zip` загрузите в приватный Kaggle Dataset и подключите к Notebook. Отдельно подключите официальный **PyTorch iResNet50/MS1MV3 checkpoint** — источник и порядок описаны в инструкции. Без корректного checkpoint запуск останавливается до подготовки данных.

## Конвейер

1. FairFace: `mehmoodsheikh/fairface-dataset`, автоматическая загрузка через kagglehub либо подключённая папка. Выбираются 5000 уникальных читаемых негативов, seed=42.
2. [Наши позитивы](data/raw/sokolov_positive_v1/README.md): 1000 кадров веб-камеры из четырёх лекций. Метки `1=sokolov`, `0=not_sokolov`.
3. Аудит исходников и точных дублей. Один случайный стратифицированный split **70/15/15** до аугментаций; обе модели используют одни назначения. Явно заданный внешний train/test сохраняется.
4. Один детектор SCRFD и выравнивание по пяти точкам для обоих классов. В обучение включаются только изображения с единственным обнаруженным лицом. Пропуски и неоднозначные случаи сохраняются в аудите и исходном test.
5. Обучение бинарных голов через BCEWithLogitsLoss. Для ResNet18 затем дообучается последний блок; ArcFace encoder по умолчанию заморожен.
6. Лучший checkpoint — по validation AP; порог — по validation F1. Test не используется в подборе параметров.
7. Выбор победителя между архитектурами по validation AP, сохранение `best_model.pt`, генерация из пяти шумов и визуализация шагов.
8. Сравнительный отчёт, артефакты и отдельная публикация.

**Ограничение выбранного протокола:** кадры одних лекций могут попасть в разные выборки; личности FairFace не гарантированно разделены. Источники позитивов и негативов различаются. Метрики случайного test могут быть завышены и не характеризуют перенос на новую лекцию.

## Команды

Из корня проекта с установленными зависимостями:

```bash
python -m sokolov check-config
python -m sokolov check-weights --arcface-checkpoint /path/to/backbone.pth
python -m sokolov run-all --run-dir /kaggle/working/runs/experiment-01 --arcface-checkpoint /path/to/backbone.pth
python -m sokolov run-all --run-dir /kaggle/working/runs/experiment-01 --arcface-checkpoint /path/to/backbone.pth --resume
python -m sokolov synthesize --run-dir /path/to/completed-run
python -m sokolov report --run-dir /path/to/completed-run
python -m sokolov publish-report --run-dir /path/to/completed-run
```

Отдельные стадии: `download-data`, `prepare`, `train`, `evaluate`, `report`. `select-best` сохраняет победителя по validation AP. `synthesize` использует `best_model.pt` без повторного обучения и исходных данных. `--config` выбирает одну модель; `run-all` и `check-weights` используют обе конфигурации из `configs/`. Публикация никогда не запускается автоматически из `run-all`.

`check-config` использует только стандартную библиотеку Python 3.11+ и не скачивает данные/веса. Проверка конфигурации не означает готовность обучения.

## Отчёт

Image-level и face-level метрики: ROC-AUC, average precision, accuracy, balanced accuracy, precision/recall/F1 по классам, macro-F1, specificity, MCC, confusion matrix, log loss, Brier score. Также сохраняются ROC/PR-кривые, история обучения, распределения оценок, FP/FN, покрытие детектора, время обучения и производительность классификатора. Генерация показывает пять финалов, промежуточные шаги и графики логита/оценки. Высокая оценка узора не гарантирует узнаваемое лицо.

Отсутствие лица и неоднозначная детекция дают отрицательное image-level решение; положительный пропуск — FN. Неопределённые показатели записываются как N/A с причиной. Checkpoints, полные предсказания и изображения ошибок остаются локальными. Команда публикации отправляет только завершённый нетехнический отчёт, агрегированные метрики и графики в `reports/<run_id>/`.

## Разработка

Python 3.11/3.12 для ML-окружения. В Kaggle устанавливайте `requirements-kaggle.txt`; версии CPU-обучения проверены локально, полный GPU-запуск остаётся пользовательским этапом.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-kaggle.txt
pip install --no-deps --no-build-isolation -e .
PYTHONPATH=src python -m unittest discover -s tests -v
```

[Результаты технической проверки](docs/VALIDATION.md), [схемы данных](data/README.md), [архитектура](outputs/architecture.md).

Код и позитивный набор v1 включены в Git. FairFace, подготовленные данные, реальные рабочие манифесты, веса и большие артефакты запусков исключены. FairFace используется с атрибуцией [авторам](https://github.com/joojs/fairface), CC BY 4.0. Предобученные веса InsightFace — для некоммерческих исследований.

## Android — камера и фото

[Инструкция для Galaxy A35](docs/ANDROID.md). Приложение в `android/` проверяет лица с камеры и выбранных фотографий, работает без сервера, позволяет переключать две модели и показывает время обработки. ZIP с обеими лучшими моделями создаётся после обучения:

```bash
python -m sokolov export-mobile --run-dir /path/to/run --output mobile_models.zip
```

Лучшие checkpoints сохраняются для каждой архитектуры, чистые веса — в best_weights.pt. Приложение не включает случайные «готовые» веса; реальная скорость 5–10 проверок/с требует измерения на телефоне. Подготовка/экспорт не заменяют обучение.
