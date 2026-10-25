# Код, который стоит читать при реализации

Проверено 2026-10-25: страницы репозиториев и перечисленные ниже файлы открыты;
установка и execution не выполнялись. Репозитории не клонировались в workspace.
Ссылки на `main` подвижны: перед заимствованием фиксировать commit.

## Основной ориентир: Snap GRID

[Авторский репозиторий](https://github.com/snap-research/GRID).
Он содержит последовательность embedding inference → SID training/inference →
generative training/inference. Используем его как reference, не выдаём за
официальный исходный код Google TIGER.

| Файл | Что смотреть |
|---|---|
| [sem_embeds_inference_flat.yaml](https://github.com/snap-research/GRID/blob/main/configs/experiment/sem_embeds_inference_flat.yaml) | Формат embedding inference и размер входного текста; по умолчанию flan-t5-xl |
| [rkmeans_train_flat.yaml](https://github.com/snap-research/GRID/blob/main/configs/experiment/rkmeans_train_flat.yaml) | Уровни, размер codebook и контракт входных embeddings |
| [tiger_train_flat.yaml](https://github.com/snap-research/GRID/blob/main/configs/experiment/tiger_train_flat.yaml) | Dataset, label masking, collate, training/validation и настройки модели |
| [tiger_inference_flat.yaml](https://github.com/snap-research/GRID/blob/main/configs/experiment/tiger_inference_flat.yaml) | SID map, encoder/decoder и число результатов генерации |
| [tiger_generation_model.py](https://github.com/snap-research/GRID/blob/main/src/models/modules/semantic_id/tiger_generation_model.py) | Base generative recommender, prefix checking, encoder–decoder и генерация |

Что потребует адаптации:

- Конфиги используют TFRecordIterator и ожидают определённую структуру полей.
  Наши pickle-таблицы baseline не являются готовым dataset для этих команд.
- В train-конфиге есть `test: true`: перед переносом проверить фактический порядок
  вызовов и отключить итоговый test при разработке. Сам по себе флаг ещё не
  доказывает утечку или подбор по тесту.
- Не копировать пути/размеры/accelerator настройки как наши готовые параметры.

[requirements.txt](https://github.com/snap-research/GRID/blob/main/requirements.txt)
собран под Python 3.10 и CUDA 12.4, включает крупный инфраструктурный стек.
Наш Python 3.12 и GPU требуют отдельной проверки совместимости минимального набора
зависимостей. Не запускать бездумно установку всего списка в рабочее окружение.
[LICENSE](https://github.com/snap-research/GRID/blob/main/LICENSE)
ограничивает использование образца некоммерческими исследовательскими целями;
при переносе кода сохранять соответствующие условия и notices.

## Независимые реализации TIGER

| Репозиторий | Проверенный по README охват | Как применять |
|---|---|---|
| [XiaoLongtaoo/TIGER](https://github.com/XiaoLongtaoo/TIGER) | RQ-VAE, генерация кодов, T5; явно unofficial | Второй reference, сверить решения с GRID и статьёй |
| [superkosat/TIGER](https://github.com/superkosat/TIGER) | RQ-VAE и decoder-only генератор | Полезен для альтернативного backbone; не смешивать с encoder–decoder абляцией |

README сообщают о поддержке этапов и результатах авторов репозиториев;
мы их не воспроизводили. Перед использованием проверить конкретный commit,
лицензию и train/evaluation код. Число звёзд не является критерием корректности.

## Основа квантования и API декодирования

- [kakaobrain/rq-vae-transformer](https://github.com/kakaobrain/rq-vae-transformer):
  авторская реализация CVPR 2022 для изображений. Читать residual quantizer,
  не переносить image pipeline в KION.
- [Hugging Face generation utilities](https://huggingface.co/docs/transformers/main/en/internal/generation_utils):
  prefix restrictions, scores, cache. Версию API закрепить вместе с кодом.

## Предлагаемая стратегия

Первую KION-версию сделать небольшой и явной: собственные time splits, готовый
замороженный multilingual encoder, residual k-means, маленький seq2seq генератор,
trie и проверяемый beam search. Сверять этапы с GRID. После end-to-end smoke
добавлять RQ-VAE. Это позволит локализовать ошибки по стадиям и не переносить
чужой data protocol вместе с моделью.
