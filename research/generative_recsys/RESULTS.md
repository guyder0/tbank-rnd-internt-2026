# Результаты экспериментов

Срез: 2026-10-25. Все приведённые генераторы оценены на **одной validation**:
3328 warm пользователей, 9083 новых положительных пар, cutoff 2021-08-09,
горизонт 7 дней. Train cutoffs: 2021-07-19, 07-26, 08-02; 30000 dev-user IDs,
seed 42, личные истории строго до cutoff. Это подбор моделей на validation,
не независимое подтверждение качества; seed sweep и новый final test не выполнены.

## Рекомендации

| Представления → tokenizer | NDCG@10 | Recall@10 | MAP@10 | Run |
|---|---:|---:|---:|---|
| Популярность полного train-каталога | 0.12330 | 0.18897 | 0.08363 | общий контроль |
| TF-IDF/SVD → residual k-means | 0.13568 | 0.19986 | 0.09512 | 163621 |
| **TF-IDF/SVD → RQ-VAE** | **0.14054** | **0.20995** | **0.09785** | **170957** |
| Sentence-T5 → RQ-VAE | 0.13651 | 0.20084 | 0.09617 | 173448 |
| E5-base → RQ-VAE | 0.13593 | 0.20333 | 0.09528 | 180753 |
| E5-base → alternating + affine RQ-VAE | 0.13512 | 0.20125 | 0.09439 | 190501 |
| Ручной CatBoost strong expanded | 0.16041 | 0.23361 | 0.11351 | внешний reference |

Генераторы используют фиксированный каталог 14947 объектов до первого cutoff;
средний ceiling recall 0.87793. Недоступные positives остаются в знаменателе.
CatBoost использует expanded candidate retrieval: это сравнение **целых систем**,
не чистое сравнение ranker на одинаковых кандидатах. Reference отличается от
combined seed44 рецепта baseline_docker; его происхождение указано в snapshots.

Выбранный для Docker рецепт: `configs/best_tfidf_rqvae.json`, run
`20261025T170957_093408Z`, лучший generator epoch=3. NDCG-дельта к популярности
+0.01724, paired user-bootstrap 95% CI [0.01286, 0.02155]. Интервал условный на
выбранной модели, не учитывает перебор гиперпараметров. Улучшение reconstruction
и количества используемых SID не привело к улучшению рекомендаций у affine.

## Изолированный RQ-VAE на E5-base

Все полные запуски ниже используют те же 14947 E5-векторов; средний вектор даёт
reconstruction MSE=0.137703. Ошибка — среднее squared L2 с суммой по координатам.
Это train reconstruction, не ranking или held-out качество.

| Режим | LR | Эпох | Reconstruction ↓ | Выигрыш к среднему | Используемые коды |
|---|---:|---:|---:|---:|---|
| joint | 0.1 | 1000 | 0.122429 | 11.09% | 3 / 52 / 56 |
| alternating | 0.1 | 1000 | 0.119462 | 13.25% | 7 / 59 / 61 |
| alternating | 0.4 | 1000 | 0.137746 | −0.03% | 1 / 1 / 1 |
| alternating | 0.1 | 200 | 0.137844 | −0.10% | 1 / 61 / 60 |
| alternating + affine | 0.1 | 1000 | 0.119599 | 13.15% | 38 / 62 / 53 |

Последняя модель имеет 8952 разных троек SID против 5428 у alternating без
Affine; это не гарантия семантической или рекомендательной полезности.
Имена экспериментальных директорий могут не соответствовать фактическим
параметрам: CSV и эта таблица используют **config.json**, а не имя папки.

## Сохранённые данные

- [results/recommendation_metrics.csv](results/recommendation_metrics.csv): точные метрики генераторов.
- [results/rqvae_metrics.csv](results/rqvae_metrics.csv): параметры и reconstruction.
- [results/experiments.json](results/experiments.json): полный индекс и SHA256 snapshots.
- `results/runs/<UTC>/`: config, metrics, curves, per-user CSV, diagnostic plots,
  manifest и снимок исходников каждого полного эксперимента (без весов/данных).
- `results/cleanup_manifest.json`: куда архивированы локальные smoke/failed проверки.

Файлы results не исключены .gitignore. Полные бинарные модели остаются в
`artifacts/`; для переноса точного лучшего генератора есть `generative_docker.export_best`.
