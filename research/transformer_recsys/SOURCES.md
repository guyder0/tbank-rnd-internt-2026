# Источники и маршрут чтения

Проверено 2026-10-24. Ниже только первичные источники. Дата препринта и дата
конференции могут отличаться. «Просмотрен» означает проверку страницы и заявленных
результатов; это не воспроизведение экспериментов. Для HSTU, OneRec и ReST дополнительно
разобраны методы в полном HTML; для YouTube/SASRec/TiSASRec — соответствующие
фрагменты авторских текстов. Полные тексты остальных работ требуют дальнейшего
предметного разбора перед реализацией их конкретных архитектур.

## Читать первыми

1. **[YouTube DNN — Covington et al., 2016](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/).**
   Откуда взялось разделение retrieval/ranking. Смотреть рисунки архитектуры
   и обсуждение feature engineering.
2. **[DIN — Zhou et al., 2017/2018](https://arxiv.org/abs/1706.06978).**
   Как кандидат определяет полезную часть истории; не Transformer.
3. **[SASRec — Kang, McAuley, 2018](https://arxiv.org/abs/1808.09781).**
   Минимальная causal self-attention модель рекомендаций.
4. **[PinnerFormer — Pancha et al., 2022](https://arxiv.org/abs/2205.04507).**
   Горизонт предсказания и стоимость пакетного профиля.
5. **[TransAct — Xia et al., 2023](https://arxiv.org/abs/2306.00248).**
   Как последовательная модель встраивается в промышленный ranking.
6. **[HSTU — Zhai et al., ICML 2024](https://arxiv.org/html/2402.17152v3).**
   §2 — постановка замены признаков, §3 — устройство и вычисления.
7. **[ReST — Chen et al., сентябрь 2026](https://arxiv.org/html/2609.01240v1).**
   Shared-prefix encoder и лёгкая обработка кандидатов. Свежий препринт;
   промышленные заявления пока не проверены независимо.
8. **[On Sampled Metrics — Krichene, Rendle, KDD 2020](https://research.google/pubs/on-sampled-metrics-for-item-recommendation/).**
   Почему нельзя брать чужие NDCG без проверки протокола.

## Исторический и методологический контекст

| Материал | Зачем нужен |
|---|---|
| [BPR, Rendle et al., UAI 2009; arXiv 2012](https://arxiv.org/abs/1205.2618) | Обучение предпочтительного порядка на implicit feedback |
| [GRU4Rec, Hidasi et al., 2015/2016](https://arxiv.org/abs/1511.06939) | История как состояние вместо статичного профиля |
| [DIEN, Zhou et al., 2018/2019](https://arxiv.org/abs/1809.03672) | Эволюция интереса и вспомогательное обучение |
| [DLRM, Naumov et al., 2019](https://arxiv.org/abs/1906.00091) | Категориальные embeddings, числовые поля, обучение взаимодействий |
| [BERT4Rec, Sun et al., 2019](https://arxiv.org/abs/1904.06690) | Masked modeling против next-item |
| [TiSASRec, Li et al., WSDM 2020, авторский PDF](https://cseweb.ucsd.edu/~jmcauley/pdfs/wsdm20b.pdf) | Время между действиями против одних позиций |
| [Set Transformer, Lee et al., ICML 2019](https://arxiv.org/abs/1810.00825) | Внимание к множествам без выдуманного порядка |
| [BERT4Rec replicability, Petrov, Macdonald, RecSys 2022](https://arxiv.org/abs/2207.07483) | Проверка реализации и достаточности обучения |
| [Huang, v2 от 15.09.2026](https://arxiv.org/abs/2504.09596v2) | Концептуальная BlueSky-критика LLM-аналогии, не новая модель |

## Длинные истории, генерация и свежая промышленная практика

| Материал | Зачем нужен / статус свидетельства |
|---|---|
| [SIM, Pi et al., 2020](https://arxiv.org/abs/2006.05639) | Поиск релевантного поднабора длинной истории; авторское описание deployment |
| [Monolith, Liu et al., 2022](https://arxiv.org/abs/2209.07663) | Реальное время и динамические embedding-таблицы |
| [TIGER, Rajput et al., NeurIPS 2023](https://arxiv.org/abs/2305.05065) | Генерация Semantic IDs вместо ANN retrieval |
| [OneRec, Deng et al., 2025](https://arxiv.org/html/2502.18965v1) | Генерация сессий и preference alignment; заявлен Kuaishou deployment |
| [SORT, Wang et al., v3 от 31.08.2026](https://arxiv.org/abs/2603.03988v3) | Промышленный ranking Transformer; заявлен AliExpress deployment |
| [TGR Team, сентябрь 2026](https://arxiv.org/abs/2609.00986v1) | Карта ranking/generation/reasoning внутри Tencent; авторский препринт |
| [Instagram model fleet, Meta, 21.05.2025](https://engineering.fb.com/2025/05/21/production-engineering/journey-to-1000-models-scaling-instagrams-recommendation-system/) | Свидетельство продолжающегося сосуществования множества моделей и стадий |
| [GEM, Meta, 13.10.2026](https://engineering.fb.com/2026/10/13/ml-applications/generative-ads-model-gem-scaling-frontier-ads-ai/) | Foundation-модель и перенос знаний; официальный инженерный отчёт |
| [HSTU serving, NVIDIA, 30.09.2026](https://developer.nvidia.com/blog/deploying-an-hstu-generative-recommender-with-nvidia-dynamo-triton/) | AOTI, KV cache и serving; демонстрация поставщика со своими условиями замера |

## Репозитории для изучения реализации

| Репозиторий | Как использовать в исследовании |
|---|---|
| [kang205/SASRec](https://github.com/kang205/SASRec) | Оригинальный TensorFlow-код; старый стек, не инструкция установки |
| [pmixer/SASRec.pytorch](https://github.com/pmixer/SASRec.pytorch) | Отдельная PyTorch-реализация; её автор не равен авторам исходного SASRec |
| [pinterest/transformer_user_action](https://github.com/pinterest/transformer_user_action) | Официальный модуль TransAct, не вся система Pinterest |
| [meta-recsys/generative-recommenders](https://github.com/meta-recsys/generative-recommenders) | HSTU и public benchmarks; не полная закрытая production-система Meta |
| [NVIDIA/recsys-examples](https://github.com/NVIDIA/recsys-examples) | Оптимизации embeddings/training/serving; сначала проверить аппаратные требования |

README этих репозиториев просмотрены; зависимости не устанавливались, код не
исполнялся. Перед заимствованием потребуется проверить лицензию, закрепить commit
и отдельно проверить протокол оценки: код из статьи не является автоматической
гарантией отсутствия утечек.

## Следующие вопросы для чтения, не закрытые этим обзором

- Как меняется качество при одинаковых loss, negatives и бюджете обучения?
- Насколько выигрыш HSTU определяется блоком, а насколько организацией данных?
- Какие сигналы глобального свежего спроса нужны модели без ручных агрегатов?
- Как лучше учить будущий недельный набор при неизвестном порядке внутри дня?
- Когда content embeddings помогают sparse/cold items при равной доступной информации?
- Насколько выводы KION сохраняются на настоящем event-log с повторными действиями?

Публикации компаний доказывают наличие публичных свидетельств конкретных подходов,
но не их распространённость на рынке и не переносимость закрытых A/B результатов.
