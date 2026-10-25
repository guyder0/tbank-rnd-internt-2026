# Исторический журнал SID-прототипа

Актуальная сводка и лучший рецепт: [RESULTS.md](../RESULTS.md).
Разделы ниже сохраняют состояние на момент соответствующих экспериментов.

Дата: 2026-10-25. Реализация находится в `run.py` и `src/`.

## Обновление: multilingual E5-base

Текущий default — `intfloat/multilingual-e5-base`, 768 dimensions, `query: `,
masked mean pooling, L2 normalization, FP32, 256 tokens, batch 16.
Параметры RQ-VAE и генератора прежние. Пройдено 26 тестов, включая ручную
проверку pooling/prefix на настоящем маленьком локальном Transformer и полный
CPU smoke до generation/save/load/metrics. Pretrained E5-base в кэше нет;
реальные веса и GPU-качество проверит пользователь. Команды — в RUNNING.md.

## Предыдущий вариант: Sentence-T5

Default заменён на `sentence-transformers/sentence-t5-base`, 768 dimensions,
полная Sentence-Transformers цепочка (pooling + Dense), FP32, 256 tokens,
batch 16, L2 normalization. RQ-VAE и генератор сохранены. Checkpoint английский,
применяется к русским метаданным как отдельный эксперимент.
25 тестов пройдены, включая настоящую локальную tiny T5 + pooling + Dense
и полный CPU smoke с ней (RQ-VAE, generator, save/load, generation, metrics).
Отдельный автономный TF-IDF smoke также пройден. Загрузка настоящих
весов в агентской среде блокируется DNS Hugging Face; качество не измерено.
Проверка с весами: `run --smoke --pretrained-smoke`, команды — в RUNNING.md.

## Обновление: RQ-VAE

По умолчанию квантователь заменён на RQ-VAE по TIGER. Пройден 21 тест и
синтетический/реальный CPU smoke. Команды, параметры и новые артефакты —
в [RQVAE.md](RQVAE.md). Разделы ниже описывают первоначальный residual k-means
прототип; его артефакты и результаты сохранены. Полное обучение нового варианта
ещё не выполнялось.

## Реализовано

- Полные временные истории выбранных пользователей KION, без будущих событий;
  недельная цель и dev-user IDs совпадают с исходным baseline.
- Закрытый каталог до первого train cutoff; lexical TF-IDF/SVD embeddings,
  residual k-means и уникальные SID с collision suffix.
- Опциональный замороженный multilingual E5 backend.
- Time-aware Transformer encoder–decoder без искусственного порядка внутри дня;
  обучение token CE, веса запросов, early stopping по end-to-end NDCG@10.
- Constrained beam search с исходными log probabilities, исключением seen,
  точный перебор для маленького каталога и контроль checkpoint reload.
- Популярность по всему прошлому каталогу и закрытому SID-каталогу; сравнение
  с сохранённым expanded CatBoost при совпадении validation truth и пользователей.
- Общие CPU/CUDA-пути, постоянные логи/статусы/метрики, хеши и копии исходников.

## Выполненные проверки

В `baseline/.venv` доступен `torch==2.7.1+cu126`; все выполненные нейросетевые
проверки использовали CPU. Видеокарта агенту недоступна.

Команда тестов:

```bash
baseline/.venv/bin/python -m unittest discover -s research/generative_recsys/tests -v
```

14 тестов проходят без пропусков: уникальность и детерминизм SID, random-code
контроль, отсечение будущего, полный учёт retrieval misses, beam/exact scoring,
целые группы дат, target causal mask, padding, перестановки одного дня,
конечные градиенты, tiny overfit и явный отказ CUDA при её отсутствии.

Проведены end-to-end CPU-запуски:

| Проверка | Данные и обучение | Результат |
|---|---|---|
| Синтетический smoke | 32 объекта, 72 training pairs, 24 validation users, 2 мини-эпохи | Все стадии, checkpoint reload и beam/exact сравнение прошли |
| Реальный smoke | KION, первые 300 dev-user IDs, 281 training pairs, 33 validation users, 2 мини-эпохи | Генерация, метрики, bootstrap и CatBoost comparison прошли |
| Основная форма модели | KION, первые 100 dev-user IDs, 131 training pairs, 12 validation users, 1 эпоха; d=128, 2 blocks, width=64 | Завершение без ошибок с основными размерами архитектуры |

Smoke-метрики SID — NDCG@10 = 0. Это **не результат обучения для качества**:
слишком мало шагов и пользователей. Преимущество над популярностью или CatBoost
не установлено. Полное обучение на CPU не запускалось.

## Подготовлено для GPU-запуска

`artifacts/kion_tfidf_prepared/` содержит полноценную подготовку всех 30 000
исходных dev-user IDs с конфигурацией `configs/default.json`:

- 14 947 объектов, наблюдавшихся до 2021-07-19;
- embeddings размерности 96; 3 codebooks × 64, все центроиды используются;
- 9 992 уникальных префикса до suffix, максимальная collision-группа 83;
  после suffix все полные SID уникальны;
- train: 2 704 / 2 886 / 3 042 запросов по трём датам;
- validation: 3 328 пользователей, 9 083 positives;
- полная подготовка заняла около 23 секунд в этом запуске.

На этой полной validation-группе популярность даёт NDCG@10 **0.123301**,
как в исходном baseline. Популярность по закрытому SID-каталогу — **0.124554**.
Это проверка согласованности метрик и отправная точка для будущего эксперимента.
Метрика `candidate_recall` при передаче evaluator только готового top-10
численно равна Recall@10; для полного доступного SID-каталога отдельно считается
`catalog_recall_ceiling`.

## Ещё не проверено

- Полное GPU-обучение, фактическая VRAM и latency CUDA.
- E5 backend с настоящими скачанными весами.
- Качество после полноценного обучения, несколько seeds и tokenizer ablations.
- Оценка двух моделей на одном фиксированном наборе кандидатов: текущий отчёт
  сравнивает целые системы с разным retrieval, что прямо указано в metrics.
- Независимый финальный test: известная test-неделя не открывается этим кодом.

Команды установки, smoke и GPU-запуска — в [RUNNING.md](../RUNNING.md).
Снимок фактических зависимостей — `requirements.lock.txt`; это не команда
безусловной переустановки всего рабочего окружения.
