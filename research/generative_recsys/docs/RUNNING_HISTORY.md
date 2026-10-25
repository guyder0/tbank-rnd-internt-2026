# Запуск первого SID-бейзлайна

Реализация по умолчанию: **content → RQ-VAE → уникальные SID → time-aware
Transformer encoder–decoder → constrained beam search**. Генератор сразу
рекомендует фильмы из каталога; предварительного популярностного/жанрового отбора
в нём нет. Это небольшая адаптация идей TIGER/GRID к недельной задаче KION.

Актуальная замена tokenizer и команда нового запуска описаны в [RQVAE.md](RQVAE.md).
Прежний residual k-means сохранён для контрольного сравнения.

## Зависимости и окружение

Используем только существующий `baseline/.venv` (Python 3.12).
Для обучения нужен PyTorch. В обычном терминале компьютера из корня проекта:

```bash
uv pip install --python baseline/.venv/bin/python 'torch==2.7.1' \
  --index-url https://download.pytorch.org/whl/cu126
```

Эта CUDA-сборка пригодна также для CPU-проверок. Для машины без GPU можно выбрать
вместо `cu126` индекс `cpu`; такая сборка не сможет потом обучать на CUDA.
Перед запуском GPU проверить:

```bash
baseline/.venv/bin/python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available())'
```

При `--device cuda` отсутствие CUDA приводит к ранней ошибке, без скрытого перехода
на CPU. В текущем окружении уже доступен `torch==2.7.1+cu126`; CPU-проверки выполнены.
GPU агенту недоступна. Фактические проверки — в `IMPLEMENTATION_STATUS.md`.

## Мини-тесты и CPU smoke

```bash
baseline/.venv/bin/python -m unittest discover -s research/generative_recsys/tests -v
baseline/.venv/bin/python -u -m research.generative_recsys.run --smoke
```

Smoke не скачивает весов и не требует исходных данных: синтетический русский
каталог, настоящее маленькое обучение на CPU, генерация, оценка, сохранение
и загрузка checkpoint. Работают те же модель и decoder, что в GPU-эксперименте.
Без PyTorch нейросетевые тесты явно пропускаются, а smoke останавливается:
такой запуск нельзя считать полной проверкой.

Короткая проверка на настоящих данных (300 dev-user IDs, вся их история):

```bash
baseline/.venv/bin/python -u -m research.generative_recsys.run --smoke --real-data
```

Проверить только подготовку/SID без PyTorch:

```bash
baseline/.venv/bin/python -u -m research.generative_recsys.run --smoke --prepare-only --tokenizer residual_kmeans
```

Метрики smoke не оценивают качество модели. Реальный smoke использует намеренно
малый codebook, поэтому collision-suffix может быть большим; это не настройки
полного эксперимента.

## Основной GPU-эксперимент: multilingual E5-base → RQ-VAE

Установить зависимости в текущее окружение:

```bash
uv pip install --python baseline/.venv/bin/python -r research/generative_recsys/requirements-sentence-t5.txt
```

Короткая CPU-проверка с настоящими весами (первый запуск скачивает модель):

```bash
baseline/.venv/bin/python -u -m research.generative_recsys.run --smoke --pretrained-smoke
```

Затем полный запуск:

```bash
baseline/.venv/bin/python -u -m research.generative_recsys.run --device cuda
```

По умолчанию замороженный `intfloat/multilingual-e5-base` кодирует русские
названия, жанры, страны, режиссёров, актёров и тип контента. Префикс `query: `
для каждого фильма, mean pooling с attention mask и L2 normalization — по
[карточке E5](https://huggingface.co/intfloat/multilingual-e5-base).
На выходе 768-мерные embeddings без SVD, максимум 256 токенов, batch 16, FP32.
Длина входа и параметры RQ-VAE/генератора сохранены относительно Sentence-T5.
Веса кэшируются в `artifacts/hf_cache/`; model revision, prefix, pooling и
параметры кодирования записываются в manifest. Новых зависимостей после
Sentence-T5 не требуется; в чистой среде достаточно `transformers>=4.51,<5`.
26 тестов пройдены, включая offline encoder/pooling и полный CPU smoke.
Настоящий pretrained E5-base пока не проверен: весов в локальном кэше нет.
Обычный `--smoke` остаётся полностью автономным TF-IDF тестом; только явный
`--pretrained-smoke` использует pretrained weights.
Все 30 000 dev-user IDs те же, что
в baseline; train cutoffs 19/26 июля и 2 августа, validation 9 августа 2021.
Коды теперь обучает RQ-VAE с бюджетом 500 эпох; детали отличий от статьи — в RQVAE.md.
Полные истории сохраняются в подготовленных данных; encoder берёт последние
128 объектов с сохранением целой пограничной даты, поэтому фактический cap мягкий.

Если папка `artifacts/kion_tfidf_prepared` из агентской подготовки существует,
можно переиспользовать её только для прежнего residual k-means варианта:

```bash
baseline/.venv/bin/python -u -m research.generative_recsys.run --device cuda \
  --tokenizer residual_kmeans \
  --config research/generative_recsys/configs/residual_kmeans.json \
  --prepared research/generative_recsys/artifacts/kion_tfidf_prepared
```

Прежний TF-IDF + RQ-VAE: добавить `--embedding-backend tfidf` без `--prepared`.
Для E5 нельзя передавать старые TF-IDF или Sentence-T5 prepared caches.

Каждый запуск создаёт новую папку `research/generative_recsys/artifacts/<UTC>/`.
Прежние модели и baseline не перезаписываются. Prepared cache проверяется по хешу
и параметрам; возобновление optimizer-state после прерывания пока не реализовано.

Чтобы закрыть терминал и продолжить обучение:

```bash
nohup baseline/.venv/bin/python -u -m research.generative_recsys.run --device cuda \
  > research/generative_recsys/artifacts/gpu_launcher.log 2>&1 &
tail -f research/generative_recsys/artifacts/gpu_launcher.log
```

Если папки `artifacts` ещё нет, предварительно создать её командой
`mkdir -p research/generative_recsys/artifacts`.

## Сравнение с Sentence-T5

Прежний Sentence-T5 доступен явно:

```bash
baseline/.venv/bin/python -u -m research.generative_recsys.run --device cuda \
  --embedding-backend sentence_t5
```

`--embedding-backend e5` теперь выбирает **base**, не прежний small.
Текстовый encoder заморожен; параметры RQ-VAE и рекомендателя прежние.
Качество сравниваем по NDCG@10, Recall@10 и MAP@10 на тех же пользователях.
Высокий cosine similarity и число групп HDBSCAN сами по себе не являются
критерием выбора embedding model.

## Генератор с сохранённым affine RQ-VAE

Подготовлен каталог `artifacts/kion_e5_affine_frozen/`: checkpoint из
`rqvae_only/20261025T185949_751066Z_alt_affine_lr01_ep1000`, E5-base,
1000 эпох, lr=0.1, alternating + affine. Коды checkpoint проверены на CPU
на точное совпадение с `codes.npy`; неизменные исходные splits и порядок item IDs.
Ни RQ-VAE, ни E5 на этом этапе заново не обучаются.

Полное обучение **только генератора**:

```bash
baseline/.venv/bin/python -u -m research.generative_recsys.run \
  --device cuda \
  --prepared research/generative_recsys/artifacts/kion_e5_affine_frozen \
  --config research/generative_recsys/artifacts/kion_e5_affine_frozen/config.json
```

GPU-настройки генератора прежние: d_model=128, по 2 encoder/decoder слоя,
до 30 эпох, early stopping=5 по validation NDCG, beam=64, top_k=10.
Меняются только SID. Результат — новая папка `artifacts/<UTC>/`, метрики
NDCG@10/Recall@10/MAP@10 и сравнение с popularity/CatBoost.

CPU smoke на настоящих замороженных SID, 12 queries на split и tiny generator:

```bash
baseline/.venv/bin/python -u -m research.generative_recsys.run \
  --device cpu --generator-smoke \
  --prepared research/generative_recsys/artifacts/kion_e5_affine_frozen \
  --config research/generative_recsys/artifacts/kion_e5_affine_frozen/config.json
```

Этот CPU smoke пройден: 92 training pairs, 12 validation users, весь каталог
14947 items, обучение/save/load/generation/evaluation. Его качество не оценивается.
Не использовать обычный `--smoke`: он меняет tokenizer config и несовместим
с замороженными SID. Исходная подготовка выполнена модулем
`prepare_frozen_rqvae`; он отказывается перезаписывать существующий output.

## Что смотреть в логах

Изолированное обучение RQ-VAE без остальных этапов (1000 эпох, lr=0.1):
см. [RQVAE_ONLY.md](RQVAE_ONLY.md), модуль `research.generative_recsys.run_rqvae`.

- `status.json`: `prepared` для подготовки, `completed` после полного обучения,
  либо `failed` с ошибкой; Python, версии библиотек, source hashes, время и RAM.
- `run.log`: стадии, шаги обучения, loss и validation по эпохам, traceback.
- `prepare_manifest.json`, `semantic_ids.csv`, `item_embeddings.npy`, `prepared.pkl`:
  источник, хеши, временные срезы, коды и полный подготовленный вход.
  При `--prepared` они находятся в исходной папке; ссылка записана в status.
- `learning_curve.json`: loss и NDCG/Recall/MAP на validation каждой эпохи.
- `model.pt`: лучшая по validation модель; `recommendations.json`: её топ-10.
- `metrics.json`: выбранная эпоха, метрики и парный bootstrap против популярности.
  Если есть совместимые сохранённые метрики CatBoost, добавляется сравнение с ним.
- `validation_*_per_user.csv`: индивидуальные метрики для разбора ошибок.

Получены ли улучшения — решаем по `metrics.json`, не по падению train loss.
Положительный `ndcg_delta` означает преимущество SID. Bootstrap CI, пересекающий
ноль, не даёт уверенного вывода о преимуществе на этой группе пользователей.
Сам выбор лучшей эпохи по validation делает эту оценку оценкой для разработки.

## Важные границы сравнения

Популярность считается за 30 прошлых дней без просмотренных объектов. Отчёт
содержит её по всему наблюдавшемуся каталогу и по закрытому SID-каталогу отдельно.
SID-каталог фиксирован на объектах, наблюдавшихся до первого train cutoff;
недельные positives вне каталога остаются в знаменателе метрик. Его потолок
покрытия записан в `catalog_recall_ceiling`.

Генератор обучается на всех in-catalog positives выбранных train-запросов,
включая те, которые могли не попасть в кандидаты бустинга. Доступный сырой
журнал общий, но фактически использованные обучающие пары могут различаться.

Сохранённый CatBoost сравнивается только при совпадении validation truth и user IDs;
это сравнение полных пайплайнов с разным retrieval, не чистая абляция ranker.
Не утверждаем, что генератор уже заменил всю информацию глобальных агрегатов.
Test/holdout не открывается. Уже известная test-неделя не становится независимой
итоговой проверкой для новой модели.
