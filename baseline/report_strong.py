"""Regenerate the tracked report from frozen experiment artifacts."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def metric_row(split, name, metrics):
    return (f"| {split} | {name} | {metrics['ndcg@10']:.5f} | {metrics['recall@10']:.5f} | "
            f"{metrics['map@10']:.5f} | {metrics['candidate_recall']:.5f} |\n")


def main():
    output = ROOT / 'artifacts/strong'
    reports = {name: read(output / path / 'metrics.json') for name,path in [
        ('fixed','fixed'), ('expanded','expanded'), ('no_co','fixed/no_co')]}
    final = read(output / 'final/metrics.json')
    selection = read(output / 'selection.json')
    primary = selection['primary_variant']
    table = '| Часть | Метод | NDCG@10 | Recall@10 | MAP@10 | Candidate recall |\n|---|---|---:|---:|---:|---:|\n'
    table += metric_row('validation','Популярность',reports['fixed']['validation']['popularity'])
    table += metric_row('validation','Item-to-item',reports['fixed']['validation']['item_to_item'])
    table += metric_row('validation','Исходный CatBoost v1',reports['fixed']['validation']['previous_catboost'])
    table += metric_row('validation','v1 на expanded-кандидатах',reports['expanded']['validation']['previous_catboost'])
    for name in ['no_co','fixed','expanded']:
        table += metric_row('validation',name,reports[name]['validation']['catboost'])
    for name in ['popularity','item_to_item','previous_catboost']:
        table += metric_row('Новые пользователи',name,final['fixed'][name])
    table += metric_row('Новые пользователи','v1 на expanded-кандидатах',final['expanded']['previous_catboost'])
    for name in ['no_co','fixed','expanded']:
        table += metric_row('Новые пользователи',name,final[name]['catboost'])
    selected_val = reports[primary]['validation']['catboost']['ndcg@10']
    previous_val = reports['fixed']['validation']['previous_catboost']['ndcg@10']
    selected_final = final[primary]['catboost']['ndcg@10']
    previous_final = final['fixed']['previous_catboost']['ndcg@10']
    lower,upper = final['primary_vs_original_pipeline']['ci95']
    full = pd.read_csv(output/'fixed/validation_catboost_per_user.csv').set_index('user_id')
    ablated = pd.read_csv(output/'fixed/no_co/validation_catboost_per_user.csv').set_index('user_id')
    difference = (full.ndcg-ablated.ndcg).to_numpy()
    if np.isnan(difference).any(): raise ValueError('Ablation populations differ')
    rng = np.random.default_rng(42)
    interval = np.quantile([rng.choice(difference,len(difference),replace=True).mean() for _ in range(1000)],[.025,.975])
    final_full = pd.read_csv(output/'final/fixed_catboost_per_user.csv').set_index('user_id')
    final_ablated = pd.read_csv(output/'final/no_co_catboost_per_user.csv').set_index('user_id')
    final_difference = (final_full.ndcg-final_ablated.ndcg).to_numpy()
    rng = np.random.default_rng(42)
    final_interval = np.quantile([rng.choice(final_difference,len(final_difference),replace=True).mean()
                                  for _ in range(1000)],[.025,.975])
    importance = pd.read_csv((output/selection['model_path']).parent/'feature_importance.csv')
    leaders='\n'.join(f"- `{row.feature}`: {row.importance:.2f}%." for row in importance.head(8).itertuples())
    costs='| Вариант | Деревьев | Обучение, выбор итерации и оценка, с | Пиковый RSS процесса, GiB |\n|---|---:|---:|---:|\n'
    for name in ['fixed','expanded','no_co']:
        report=reports[name]
        costs+=f"| {name} | {report['trees']} | {report['train_seconds']:.1f} | {report['peak_rss_mib']/1024:.2f} |\n"
    text=f'''# Усиленный baseline на ручных агрегациях

## Результат

Основной вариант **{primary}** выбран по validation до итоговой оценки.
NDCG@10 на validation: {selected_val:.5f} против {previous_val:.5f} у v1
(прирост {(selected_val/previous_val-1)*100:.2f}%). На отложенных пользователях:
{selected_final:.5f} против {previous_final:.5f} (прирост {(selected_final/previous_final-1)*100:.2f}%).
95% парный bootstrap-интервал итоговой разницы: [{lower:.5f}, {upper:.5f}].

## Постановка и протокол

Русскоязычный KION из полного `data_original.zip`. Цель — новые наблюдаемые пары
user–item в следующие семь дней. 30 000 пользователей для разработки, seed 42.
Train: 19 и 26 июля, 2 августа; validation: 9 августа. Для итоговой недели
16–23 августа заранее выбраны ещё 30 000 пользователей, seed 20261023,
не входящих в пользователей обучающих и валидационных пар. В итоговой метрике {final['users']} пользователей
с будущими положительными объектами. Прошлая история всех пользователей доступна
всем подходам; будущие итоговые исходы не используются до фиксации моделей.

Это новые пользователи в **той же уже известной календарной неделе**, а не новый
временной тест. Различие с исходным тестом v1 — состав пользователей, поэтому
здесь v1 заново оценивается на той же группе, что и новые варианты.

## Ручные признаки и кандидаты

74 исходных + 81 дополнительный вручную заданный признак. Обучается только
CatBoost; дополнительных обучаемых представлений нет. Новые группы:

- Совместные просмотры объектов: нормированные счётчики пересечения пользователей,
  сглаженные по поддержке; сумма, максимум, число соседей, последние 1/3/5 пар,
  затухание 7/30 дней. 150 соседей объекта, поддержка от 3 пользователей.
- Групповая популярность за 7/30 дней: возраст, пол, дети, доход, возраст×пол;
  счётчик, сглаженная частота и отношение к глобальной популярности.
- Свежесть спроса: 1/3/14 дней, ускорение; активные дни пользователя и интервалы.
- Предпочтения по актёрам, студиям, типу контента и возрастному рейтингу;
  общие и недавние частоты, временное затухание, совпадения с последними парами.

`fixed` использует прежние кандидаты. `expanded` добавляет до 100 соседей
просмотренных объектов, 40 популярных объектов за три дня и 30 популярных
в возрастной группе. Объединение дедуплицируется, просмотренные объекты удаляются;
будущие положительные объекты специально не добавляются.
`no_co` — прежние кандидаты и 146 признаков, без девяти co-view агрегатов.
Формулы и ограничения вычислений — в `STRONG_PLAN.md`.

## Метрики

{table}
Макроусреднение по всем тёплым пользователям с будущими положительными объектами;
промахи генерации кандидатов остаются в знаменателе. Для экономии вычислений
пользователи без будущих положительных объектов не материализуются: они и в v1
не входили в метрику. Этот фильтр не используется как признак или продуктовое правило.

Число деревьев выбирается по отчётной end-to-end NDCG@10 на validation,
а не по нормированной только на кандидатов метрике внутри CatBoost.
Проверяются каждые 25 деревьев и последняя итерация доступного обучения.

## Анализ

На одинаковых кандидатах разница `fixed - no_co` по validation составляет
{difference.mean():.5f}; 95% парный bootstrap-интервал [{interval[0]:.5f}, {interval[1]:.5f}].
Это отдельная абляция co-view при одинаковых параметрах и процедуре выбора деревьев.
На отложенных пользователях разница той же абляции: {final_difference.mean():.5f},
95% интервал [{final_interval[0]:.5f}, {final_interval[1]:.5f}].
Сравнение с v1 включает также изменения регуляризации и выбора итерации, поэтому
весь прирост относительно v1 нельзя приписать только новым признакам.

При смене только кандидатов прежняя модель на итоговой группе получает
{final['expanded']['previous_catboost']['ndcg@10']:.5f} против {previous_final:.5f}.
Полная усиленная модель получает {selected_final:.5f}: расширенный список
полезен в сочетании с новыми признаками и обучением на расширенных парах.

Наибольшая важность PredictionValuesChange у выбранной модели:

{leaders}

Важности сами по себе не доказывают причинный вклад. Подбор проводился только
по validation; три зафиксированных варианта показаны на итоговой группе для
анализа, основной вариант не выбирается повторно по её результатам.
Bootstrap: 1000 повторов по пользователям, seed 42. Интервалы относятся к одной
неделе и фиксированным моделям, не учитывают вариативность недель или обучения.

## Затраты и воспроизведение

{costs}
Обучения частично выполнялись параллельно при высокой общей загрузке машины;
время отражает этот запуск и не является контролируемым сравнением алгоритмов.
Память указана для отдельного процесса. Используются четыре потока на модель.
Признаки expanded и итоговый скоринг всех вариантов вычисляются порциями
по 500 пользователей.

Все команды и конфигурация — в `STRONG_PLAN.md`, окружение — только
`baseline/.venv`, версии — `requirements.lock.txt`. 15 тестов проходят,
включая ручную сверку формул, временное отсечение, резервирование пользователей,
точность метрики и полный маленький сценарий итоговой оценки.

Артефакты в `artifacts/strong/`: `selection.json`, `{selection['model_path']}`,
`final/metrics.json`, индивидуальные метрики и таблицы скоринга. SHA256 моделей
зафиксированы до доступа к итоговым меткам. Артефакты исключены из Git;
численные результаты сохранены в этом отчёте и воспроизводятся кодом.
Пересоздать отчёт: `baseline/.venv/bin/python -m baseline.report_strong`.

## Ограничения

Исходные строки агрегированы по user–item с датой последнего просмотра:
достоверную последовательность сеансов восстановить нельзя. Длительность и
процент просмотра не используются. Метаданные считаются статичными; временные
равенства разрешаются через item_id. Отсутствие будущей записи не означает
подтверждённый отказ. Полная история пользователей сохраняется, но co-view граф
использует не более 100 последних доступных пар каждого пользователя.

Итоговая проверка не подтверждает устойчивость на новых временных периодах;
подбор выполнен на одной валидационной неделе. Следующие исследовательские
подходы должны использовать тот же доступ к прошлым данным, кандидатов и
протокол оценки; доработка по итоговым меткам недопустима.
Цитирование датасета — по `../KION_DATASET/README.md`.
'''
    (ROOT/'STRONG_RESULTS.md').write_text(text)
    print('Written',ROOT/'STRONG_RESULTS.md')


if __name__=='__main__':
    main()
