# Замена квантователя на RQ-VAE

Обновление 2026-10-25. По умолчанию `run.py` теперь выбирает RQ-VAE.

Позднейшее обновление того же дня: default content encoder заменён на
`sentence-transformers/sentence-t5-base` (768 dimensions). Таблица ниже описывает
предыдущий измеренный TF-IDF эксперимент; генератор и параметры RQ-VAE сохранены.
Текущие команды установки и проверки — в [RUNNING.md](../RUNNING.md).

## Что реализовано по TIGER

Источник: [TIGER §3.1, RQ-VAE и Implementation Details](https://arxiv.org/html/2305.05065v3#S3.SS1).
Модуль — [src/rqvae.py](../src/rqvae.py).

Encoder переводит вектор объекта в latent representation; три обучаемых
codebook последовательно квантуют остатки. Выбранные векторы суммируются,
decoder восстанавливает исходный embedding. Функция потерь:

`L = ||x - decoder(sum(e))||² + sum_l ||sg(r_l)-e_l||² + beta*||r_l-sg(e_l)||²`.

Здесь усреднение по объектам, squared L2 суммируется по координатам. `beta=0.25`.
Straight-through передаёт reconstruction-gradient в encoder; stop-gradient
отделяет обучение codebooks от commitment. Остатки между уровнями не нормируются.
K-means используется только для инициализации codebooks на первом случайном
батче, а не как финальный способ получения SID. После обучения веса замораживаются,
весь каталог кодируется заново; suffix устраняет коллизии, как в TIGER.

## Параметры и отличие от полного воспроизведения статьи

| Настройка | Наш новый default | TIGER |
|---|---|---|
| Encoder hidden | 512 → 256 → 128 | 512 → 256 → 128 |
| Latent dimension | 32 | 32 |
| Уровни квантования | 3 | 3 |
| Codebook width | 64 | 256 |
| Optimizer | Adagrad | Adagrad |
| Learning rate | 0.05 | 0.4 |
| Batch size | 1024 | 1024 |
| Epoch budget | 500 | 20 000 |
| Входные embeddings | Прежние TF-IDF/SVD, 96 | Sentence-T5, 768 |

Мы меняем конкретный квантователь. Сохраняем прежние content embeddings,
64 центра и настройки рекомендателя для сравнения с уже измеренным результатом.
PyTorch Adagrad initial accumulator — 0.1. Tokenizer получает только контент
каталога до первого train cutoff, без recommendation labels. Checkpoint — финальная
эпоха заданного бюджета; NDCG не используется для его выбора.

Это реализация механизма RQ-VAE из статьи, а не репликация всего TIGER.
`configs/tiger_tokenizer.json` сохраняет width/lr/epoch budget ближе к статье,
но контентный encoder, данные и рекомендатель всё равно наши. Этот более дорогой
preset не обучался полностью и не является рекомендуемым первым запуском.

## Запуск на GPU

Из корня проекта в обычном терминале:

```bash
baseline/.venv/bin/python -u -m research.generative_recsys.run --device cuda
```

Сначала заново подготавливаются embeddings и обучается RQ-VAE, затем новый
рекомендательный Transformer учится на новых SID. **Не добавлять прежний
`--prepared .../kion_tfidf_prepared`: эта папка содержит старые k-means SID.**
Скрипт проверяет тип/параметры tokenizer и отклоняет несовместимый кэш.
Исходные артефакты не перезаписываются.

Новые файлы в папке запуска:

- `tokenizer/model.pt`: encoder, decoder, codebooks и настройки RQ-VAE;
- `tokenizer/learning_curve.json`: reconstruction, codebook, commitment, total loss;
- `tokenizer/metrics.json`: использование каждого codebook, reconstruction до/после,
  фактический бюджет, проверка checkpoint reload;
- `semantic_ids.csv`, `prepared.pkl`: окончательные замороженные SID;
- корневые `metrics.json`, `run.log`: результаты рекомендаций и весь журнал.

При использовании менее 80% кодов выводится диагностическое сообщение:
малая reconstruction loss сама по себе не доказывает качество SID.

Старый вариант остаётся доступен явно:

```bash
baseline/.venv/bin/python -u -m research.generative_recsys.run --device cuda \
  --tokenizer residual_kmeans \
  --config research/generative_recsys/configs/residual_kmeans.json \
  --prepared research/generative_recsys/artifacts/kion_tfidf_prepared
```

## Что проверил агент на CPU

- 21 тест: формулы квантования/loss по ручному расчёту, направления градиентов,
  снижение reconstruction на tiny-fixture, уникальность SID, checkpoint reload,
  запрет несовместимого prepared cache и прежние тесты рекомендателя.
- Полный синтетический smoke с RQ-VAE и генератором.
- Настоящий KION: 14 947 объектов, основная RQ-сеть 512/256/128/32 и 3×64 codebooks,
  четыре optimizer steps; затем короткое обучение рекомендателя на 100 dev-user IDs,
  генерация и сравнение с популярностью/CatBoost. Завершилось без ошибок.
- На этой короткой проверке reconstruction снизилась с 1.0213 до 0.8323;
  все 64 кода каждого уровня использовались. Это не оценка качества рекомендаций.

Повторить:

```bash
baseline/.venv/bin/python -m unittest discover -s research/generative_recsys/tests -v
baseline/.venv/bin/python -u -m research.generative_recsys.run --smoke
baseline/.venv/bin/python -u -m research.generative_recsys.run --device cpu \
  --config research/generative_recsys/configs/rqvae_cpu_check.json --max-users 100
```

Полное RQ-VAE/GPU-обучение агент не запускал. E5 backend остаётся опциональным
и не проверен на скачанных весах.
