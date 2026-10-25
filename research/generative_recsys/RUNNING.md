# Запуск и воспроизводимость

Все команды выполняются из корня репозитория, в текущем Python 3.12 окружении.
Для train-once контейнера с внешним test см. [generative_docker](../../generative_docker/README.md).

## Лучший измеренный рецепт: TF-IDF → RQ-VAE → generator

```bash
baseline/.venv/bin/python -u -m research.generative_recsys.run --device cuda \
  --config research/generative_recsys/configs/best_tfidf_rqvae.json
```

Полностью локальный рецепт без pretrained downloads. Заново создаёт SID и
обучает generator на прежних KION cutoffs; это не внешний test Docker-контракта.
Логи, fingerprint, config, checkpoint и NDCG/Recall/MAP сохраняются в новом
`artifacts/<UTC>/`. CPU smoke: та же команда с `--device cpu --smoke`.

## Продолжение исследовательских вариантов

Default `configs/default.json` остаётся E5-base, **не лучшим рецептом**.
Для запуска с ним используйте `run --device cuda` без config. Для генератора с
готовым affine RQ-VAE:

```bash
baseline/.venv/bin/python -u -m research.generative_recsys.run --device cuda \
  --prepared research/generative_recsys/artifacts/kion_e5_affine_frozen \
  --config research/generative_recsys/artifacts/kion_e5_affine_frozen/config.json
```

CPU check этого cache: заменить device на cpu и добавить `--generator-smoke`.
Квантователь при --prepared не обучается. Нельзя смешивать разные embedding
backend/tokenizer/affine config с несовместимым prepared cache.

Изолированный RQ-VAE, включая lr/epochs/affine и сравнение запусков:
[docs/RQVAE_ONLY.md](docs/RQVAE_ONLY.md).

```bash
baseline/.venv/bin/python -u -m research.generative_recsys.run_rqvae \
  --device cuda --optimization alternating --affine --lr 0.1 --epochs 1000
baseline/.venv/bin/python -m research.generative_recsys.tools.summarize_rqvae
```

Диагностические tools перенесены в `research.generative_recsys.tools.*`.
Для подготовки frozen cache: `tools.prepare_frozen_rqvae`, для кластеризации —
`tools.diagnose_clusters` и `tools.compare_embedding_clusters`.

## Проверки и результаты

```bash
baseline/.venv/bin/python -m unittest discover -s research/generative_recsys/tests -v
baseline/.venv/bin/python -m unittest discover -s generative_docker/tests -v
```

Сводка экспериментов — [RESULTS.md](RESULTS.md), исходные отчёты доступны
в `results/runs/<id>/` без восстановления игнорируемых каталогов. Тяжёлые модели,
полные данные и кэши не включаются в Git. Snapshot результатов фиксирует прошлые
эксперименты; новые локальные запуски по-прежнему пишутся в artifacts.
Исторические пошаговые команды сохранены в [docs/RUNNING_HISTORY.md](docs/RUNNING_HISTORY.md).
