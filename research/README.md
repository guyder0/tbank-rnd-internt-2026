# Исследование обучаемых представлений для рекомендаций

Цель — проверить, могут ли модели, работающие с историей пользователя и контентом
фильмов, конкурировать с CatBoost на ручных агрегатах.

## Что реализовано

[Генеративный рекомендатель](generative_recsys/README.md) кодирует фильмы
Semantic IDs и предсказывает их по истории. Проверены TF-IDF/SVD, Sentence-T5,
multilingual E5-base, residual k-means и несколько режимов обучения RQ-VAE.

Лучший результат на validation: **NDCG@10 0.14054, Recall@10 0.20995,
MAP@10 0.09785** у TF-IDF/SVD + RQ-VAE. Ручной CPU baseline получает
NDCG@10 0.16041. Генеративная модель обходит популярность, но пока не CatBoost.

- [Результаты и сравнения](generative_recsys/RESULTS.md).
- [Как воспроизвести](generative_recsys/RUNNING.md).
- [Docker с обучением и внешним test](../generative_docker/README.md).
- [Общее описание проекта и future work](../README.md).

## Материалы

[Обзор трансформеров в рекомендательных системах](transformer_recsys/README.md)
содержит историю подходов, индустриальные публикации и первоначальный план
reranking-эксперимента. Сам reranker из этого плана не реализован: практические
эксперименты находятся в `generative_recsys/`.

В `generative_recsys/results/` доступны компактные отчёты без скачивания весов.
Большие модели, prepared datasets и журналы хранятся локально в `artifacts/`.
