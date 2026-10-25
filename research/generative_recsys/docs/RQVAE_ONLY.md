# Изолированная диагностика RQ-VAE

## Свои эксперименты

```bash
baseline/.venv/bin/python -u -m research.generative_recsys.run_rqvae \
  --device cuda --optimization alternating --lr 0.1 --epochs 5000 --name alt_lr01_ep5000
```

`--lr` — короткая форма `--learning-rate`. Доступны также `--seed 42`,
`--batch-size 1024`, `--optimizer adagrad` (или `adam`), `--diagnostics-every 10`.
`--optimization joint` включает прежнюю совместную оптимизацию. Без флага
optimization по умолчанию остаётся joint; указывайте alternating явно.
Каждый запуск обучается заново. `--epochs` задаёт полный бюджет этого запуска,
не добавочные эпохи к существующей модели.

Папки автоматически получают UTC и имя эксперимента; без `--name` в имени
сохраняются mode/lr/epochs/seed. Даже при одинаковом имени файлы не перезаписываются.
Есть `summary.json` с фактическими параметрами, ошибкой, gain и занятостью кодов.
Остальные подробные логи и графики сохраняются как прежде.

Сводка всех запусков, включая старые и незавершённые:

```bash
baseline/.venv/bin/python -m research.generative_recsys.tools.summarize_rqvae
baseline/.venv/bin/python -m research.generative_recsys.tools.summarize_rqvae --output /tmp/rqvae_runs.csv
```

Для сравнения проверять одинаковые seed, input_sha256, batch size и optimizer.
Smoke/limited CPU проверки отмечены отдельно; их нельзя сравнивать с полным GPU
training. Лучше менять по одному параметру. После экспериментов агент может
прочитать папки напрямую, передавать файлы вручную не требуется.

Запуск использует сохранённые эмбеддинги E5-base из
`artifacts/20261025T180753_899640Z/item_embeddings.npy`: модель E5 не запускается
повторно, рекомендательный Transformer не обучается и рекомендации не генерируются.
Исходные артефакты не меняются. Хеш embeddings сверяется с manifest.

## Эксперимент: 1000 эпох, learning rate 0.1

Из корня репозитория:

```bash
baseline/.venv/bin/python -u -m research.generative_recsys.run_rqvae --device cuda
```

Параметры по умолчанию: 1000 эпох, Adagrad lr=0.1, initial accumulator=0.1,
batch=1024, beta=0.25, seed=42, три словаря по 64 центра. Архитектура
и остальные параметры берутся из исходного E5 run, encoder/decoder/codebooks
инициализируются заново. Это не продолжение прежнего checkpoint.
Другой источник: `--prepared <папка_запуска_с_item_embeddings.npy>`.
Бюджет: `--epochs 1000 --learning-rate 0.1 --diagnostics-every 10`.

## Поочерёдная оптимизация: 1000 эпох, lr=0.1

```bash
baseline/.venv/bin/python -u -m research.generative_recsys.run_rqvae --device cuda --optimization alternating
```

Режим адаптирует §4.2, Eq.17–18 из
[Huh et al., ICML 2023](https://proceedings.mlr.press/v202/huh23a/huh23a.pdf)
к нашему residual quantizer. На каждом том же mini-batch:

1. Encoder фиксирован, обновляются только codebooks по codebook loss.
2. Заново вычисляются назначения с обновлёнными центрами. Codebooks фиксированы;
   encoder и decoder обновляются по **reconstruction loss**, без commitment
   в сетевом шаге, как в Eq.18. STE передаёт reconstruction gradient в encoder.

Два отдельных Adagrad optimizer, оба lr=0.1, initial accumulator=0.1; по одному
внутреннему/внешнему шагу. Данные, инициализация, порядок batches и 1000 эпох
сохранены. Каждый параметр обновляется один раз на batch в обоих режимах, но
чередование требует дополнительных forward и двух optimizer calls вместо одного.
`steps` — batches; `optimizer_updates` — фактические вызовы optimizer.step.
`total` в диагностике остаётся суммой трёх потерь для сопоставимости; в alternating
это не единая оптимизируемая функция. Commitment записывается диагностически.
По умолчанию без флага сохраняется прежний `joint` режим. EMA, resets, affine
parameterization и AE warm-up не добавлены. Обучение идёт заново.

CPU-проверка нового режима на полном каталоге:

```bash
baseline/.venv/bin/python -u -m research.generative_recsys.run_rqvae --device cpu --optimization alternating --max-steps 4
```

## Общее преобразование словарей: affine reparameterization

```bash
baseline/.venv/bin/python -u -m research.generative_recsys.run_rqvae \
  --device cuda --optimization alternating --affine --lr 0.1 --epochs 1000 \
  --name alt_affine_lr01_ep1000
```

Реализован **обучаемый** вариант §4.1 Eq.15 Huh et al., не EMA matching:
`effective_center[l,k] = shift[l] + scale[l] * base_center[l,k]`.
Shift и scale — 32-мерные векторы, общие для 64 центров отдельного уровня.
Начальные shift=0, scale=1 сохраняют прежнюю k-means-инициализацию и назначения.
Это прямой affine scale, без ограничения положительности.

Расстояния, суммы, остатки и экспорт центров используют effective centers.
Невыбранные base centers не получают прямого assignment gradient, но их
эффективные значения меняются через общие shift/scale. В alternating все
три набора параметров входят в codebook optimizer и фиксируются на сетевом шаге.
LR одинаковый, без дополнительного множителя для affine-параметров.

`--affine` включается явно; старые режимы и checkpoints сохраняются.
Resolved config, checkpoint settings, summary и CSV содержат affine-флаг.
Значения scale/shift записываются в `tokenizer/diagnostics.json`.
Модель E5, архитектура AE, исходные данные и остальные настройки не меняются.
Проверять эффект по реконструкции и динамике; метод не гарантирует заполнение
всех центров. Перезапуск без `--affine` даёт контроль.

## Артефакты запуска

Новая папка `artifacts/rqvae_only/<UTC>/` содержит:

- `run.log`, `status.json`, `config.json`, `input_manifest.json`, `source/`:
  журнал, статус, resolved config, хеши данных/кода, версии зависимостей;
- `tokenizer/learning_curve.json`: minibatch training losses каждой эпохи;
- `tokenizer/diagnostics.json`: полный каталог до обучения, после первой эпохи,
  далее каждые 10 эпох и в конце. Отдельные reconstruction/codebook/commitment
  losses; ошибка среднего вектора; относительный выигрыш над ним;
  дисперсия входа, latent и восстановления; использование/гистограммы кодов;
- `dynamics.png`: графики reconstruction, занятости словарей, дисперсий и потерь;
- `tokenizer/model.pt`, `tokenizer/metrics.json`, `codes.npy`:
  финальная модель, итоговые показатели, три raw кода каждого объекта.

`reconstruction_gain_vs_mean = 1 - reconstruction_mse / mean_baseline_mse`:
0 — уровень среднего вектора, положительное значение — лучше среднего.
Это реконструкция обучающего каталога, не held-out метрика и не качество
рекомендаций. Дисперсии разных пространств нельзя напрямую сравнивать как
меру потери информации; важна динамика каждого показателя.

## CPU проверки

```bash
baseline/.venv/bin/python -m unittest discover -s research/generative_recsys/tests -q
baseline/.venv/bin/python -u -m research.generative_recsys.run_rqvae --smoke
baseline/.venv/bin/python -u -m research.generative_recsys.run_rqvae --device cpu --max-steps 4
```

Последняя команда проверяет настоящий каталог и основную архитектуру, но
ограничивает optimizer четырьмя шагами. Полный CPU training агент не выполняет.
