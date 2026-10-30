# Как нейросеть представляет Евгения Соколова

Учебный бинарный классификатор: **Соколов / не Соколов**. Обе модели обучаются только на выделенных и выровненных лицах: **ResNet18** (ImageNet, 224×224) и **ArcFace iResNet50** (MS1MV3, 112×112).

Реализованы загрузка FairFace, объединение с позитивами, аудит и разбиение, SCRFD и выравнивание, обучение с продолжением, оценка, сравнительный Markdown/HTML-отчёт и отдельная публикация отчёта в GitHub. Синтез из шума остаётся будущим этапом.

**Готовых измерений качества нет.** Локальные проверки используют искусственные изображения и случайные веса; полное обучение выполняется пользователем на Kaggle GPU.

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
7. Сравнительный отчёт, артефакты и отдельная публикация.

**Ограничение выбранного протокола:** кадры одних лекций могут попасть в разные выборки; личности FairFace не гарантированно разделены. Источники позитивов и негативов различаются. Метрики случайного test могут быть завышены и не характеризуют перенос на новую лекцию.

## Команды

Из корня проекта с установленными зависимостями:

```bash
python -m sokolov check-config
python -m sokolov check-weights --arcface-checkpoint /path/to/backbone.pth
python -m sokolov run-all --run-dir /kaggle/working/runs/experiment-01 --arcface-checkpoint /path/to/backbone.pth
python -m sokolov run-all --run-dir /kaggle/working/runs/experiment-01 --arcface-checkpoint /path/to/backbone.pth --resume
python -m sokolov publish-report --run-dir /path/to/completed-run
```

Отдельные стадии: `download-data`, `prepare`, `train`, `evaluate`, `report`. `--config` выбирает одну модель; `run-all` и `check-weights` используют обе конфигурации из `configs/`. Публикация никогда не запускается автоматически из `run-all`.

`check-config` использует только стандартную библиотеку Python 3.11+ и не скачивает данные/веса. Проверка конфигурации не означает готовность обучения.

## Отчёт

Image-level и face-level метрики: ROC-AUC, average precision, accuracy, balanced accuracy, precision/recall/F1 по классам, macro-F1, specificity, MCC, confusion matrix, log loss, Brier score. Также сохраняются ROC/PR-кривые, история обучения, распределения оценок, FP/FN, покрытие детектора, время обучения и производительность классификатора.

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
