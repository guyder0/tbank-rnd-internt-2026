# Генеративный SID-рекомендатель в Docker

Контейнер воспроизводит лучший измеренный генеративный рецепт проекта:
**TF-IDF/SVD → RQ-VAE → Transformer**. Первый запуск обучает модель и оценивает
внешний test, повторные используют сохранённые веса и таблицу SID.

На исторической validation получены **NDCG@10 0.14054, Recall@10 0.20995,
MAP@10 0.09785**. Это выше популярности, но ниже ручного baseline.
Более поздние E5/affine-варианты в образ не включены: их рекомендации оказались
хуже. [Результаты и сравнения](../research/generative_recsys/RESULTS.md).

## Быстрый запуск

Окружение и общий dataset подготавливаются по [корневому README](../README.md).
Команды выполняются из корня проекта.

```bash
./generative_docker/build.sh
SID_GPU=gpu ./generative_docker/run.sh "$HOME/kion-recsys-data" \
  --inference-device cuda
```

Без GPU:

```bash
SID_GPU=cpu ./generative_docker/run.sh "$HOME/kion-recsys-data"
```

Без `SID_GPU` доступность GPU определяется автоматически. Имя образа можно
изменить через `SID_IMAGE`. По умолчанию внешняя генерация выполняется на CPU;
`--inference-device cuda` явно включает GPU и для неё.

## Данные и сохранённая модель

Входной CSV-контракт совпадает с [baseline_docker](../baseline_docker/README.md):
`train.csv`, `test.csv`, `users.csv`, `items.csv`, желательно `dataset.json`.
**Дополнительно нужен title в items.csv.** Общая утилита `prepare_kion` сохраняет
его автоматически. Пользовательская демография в генератор не подаётся.

Контейнер пишет в mounted-директорию:

```text
sid_model.pt          # generator и замороженная таблица item_id → SID
sid_model.json        # config, SHA256, происхождение и граница обучения
sid_metrics.json      # последний успешный внешний test
runs/sid_<UTC>/       # logs, config, inputs, metrics, per-user CSV, рекомендации
```

Эти имена не конфликтуют с `model.cbm/model.json` CatBoost: можно использовать
один набор входных данных для обоих решений.

При повторном запуске нужны обе model files. Если одна отсутствует, хеш неверен,
настройки несовместимы или test пересекается с периодом обучения, запуск
останавливается. Существующие модели не перезаписываются и незаметно не
переобучаются. Модель сохраняется перед оценкой test; неудачная оценка не
заменяет последний успешный отчёт. Запись защищена lock.

## Рецепт и оценка

- TF-IDF/SVD размерности 96 по названиям, жанрам, странам, режиссёрам, актёрам
  и типу контента. Скачивание языковой модели не требуется.
- RQ-VAE: encoder 96→512→256→128→32, три словаря по 64 центра, зеркальный decoder;
  500 эпох, Adagrad, lr=0.05, beta=0.25. Четвёртый код устраняет коллизии.
- Transformer: d_model=128, heads=4, по два слоя encoder/decoder, до 30 эпох,
  patience=5 по validation NDCG@10, beam=64. Исторические SID фильма суммируются
  в один вектор и дополняются давностью просмотра.

Последняя неделя train используется для validation, перед ней — три обучающих
недельных среза. До 30000 train-users выбираются с seed 42. Истории сохраняются
полностью, encoder cap=128 оставляет всю пограничную дату. После выбора модели
refit не выполняется.

Каталог фиксируется до первого train cutoff. Для тёплых пользователей генерация
идёт по его SID с исключением просмотренного. Cold users получают популярность
из train. Новые положительные фильмы вне каталога остаются в знаменателе метрик.
Test-исходы не используются при обучении и не добавляются в историю.

Отчёт содержит NDCG/Recall/MAP при K=1/5/10/20, Precision@10, HitRate@10,
overall/warm/cold, группы длины истории, сравнение с популярностью и bootstrap-
интервалы. `catalog_recall` — доля доступных в SID-каталоге positives; это не
candidate recall бустинга. Для cold-политики доступен полный train-каталог.

Параметры меняются JSON-подмножеством `default_config.json`:

```bash
./generative_docker/run.sh "$HOME/kion-recsys-data" --config /data/config.json
```

У сохранённой модели config переиспользуется; другой recipe отклоняется.
Для нового обучения выберите другую директорию модели. Новый train-сэмпл и
внешняя test-неделя могут отличаться от исторической validation, поэтому
воспроизводится рецепт, а не гарантированное число из таблицы.

## Экспорт точных лучших весов

Если доступны локальные исследовательские артефакты, до первого запуска:

```bash
baseline/.venv/bin/python -m generative_docker.export_best \
  --data-dir "$HOME/kion-recsys-data"
./generative_docker/run.sh "$HOME/kion-recsys-data"
```

Переносятся generator из `20261025T170957_093408Z` и его SID, без переобучения.
Отдельный RQ-VAE для предсказания не нужен. Экспорт не перезаписывает модель.
Эти веса не входят в Git/образ и не обучались на смонтированном train.csv;
это отмечается в metadata. Они использовали validation до 2021-08-16,
поэтому более ранний test запрещён.

## Проверить образ на маленьком примере

Создайте отдельную директорию с тестовыми данными и короткой конфигурацией:

```bash
SID_SMOKE_DIR="$(mktemp -d /tmp/kion-sid-smoke.XXXXXX)"
baseline/.venv/bin/python - "$SID_SMOKE_DIR" <<'PY'
import sys
from pathlib import Path
from generative_docker.tests.test_app import fixture
fixture(Path(sys.argv[1]))
PY

SID_GPU=cpu ./generative_docker/run.sh "$SID_SMOKE_DIR" --config /data/config.json
SID_GPU=cpu ./generative_docker/run.sh "$SID_SMOKE_DIR"
```

В первом `runs/sid_<UTC>/metrics.json` должно быть `model_action=trained`,
во втором — `loaded`. `recommendations.csv` и SHA256 модели должны совпасть.
Метрики на этих искусственных фильмах не оценивают качество рекомендаций.

Проверки реализации без Docker:

```bash
baseline/.venv/bin/python -m unittest discover -s generative_docker/tests -v
```

## Сборка и зависимости

Образ содержит только необходимые исходники, без датасета, весов и кэшей.
`source_manifest.json` фиксирует хеши кода. Версии Python 3.12.14, PyTorch
2.7.1+cu126 и транзитивных Python-зависимостей закреплены в Dockerfile/lock.
Для побайтово одинаковой сборки потребуется также закрепить digest базового
образа и Debian snapshot; CPU/CUDA могут давать численно разные результаты.

При ошибке доступа к Docker daemon проверьте возможность выполнить `docker info`
из своего терминала. Если не работает GPU passthrough, проверьте NVIDIA Container
Toolkit. Автоматический режим выбирает CPU при недоступности GPU, но ошибки
основного обучения не маскирует повторным запуском. На больших test CPU beam
search может быть медленным — используйте `--inference-device cuda` при наличии GPU.
