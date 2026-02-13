#sripts.py
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt
import numpy as np
today = date.today().strftime("%Y-%m-%d")

# =========================
# Visualization
# =========================


def plot_pie(ax, data, title):
    """
    Строит круговую диаграмму распределения значений по категориям.

    Функция принимает агрегированные данные в виде pandas Series,
    отфильтровывает нулевые значения и отображает их в виде pie chart.
    Если после фильтрации данных не остаётся, график не строится,
    а отображается заголовок с пометкой об отсутствии данных.

    Параметры
    ----------
    ax : matplotlib.axes.Axes
        Ось matplotlib, на которой строится диаграмма.
    data : pandas.Series
        Агрегированные данные, где:
        - индекс — категории (подписи секторов),
        - значения — числовые значения для построения диаграммы.
    title : str
        Заголовок диаграммы.

    Возвращает
    ----------
    None
        Функция ничего не возвращает, только строит график.
    """
    data = data[data > 0]

    if data.empty:
        ax.set_title(f'{title}\n(нет данных)')
        return

    ax.pie(
        data,
        labels=data.index,
        autopct='%1.1f%%',
        startangle=90
    )
    ax.set_title(title)



# =========================
# Sorting
# =========================


def get_ops_for_period(df, car_ids=None, date_from=None, date_to=None, op_types=None):
    """
    Фильтрует операции по автомобилям, периоду и типам операций.

    Используется для получения подвыборки данных из общего DataFrame
    перед агрегацией или визуализацией.

    Параметры
    ----------
    df : pandas.DataFrame
        Исходный DataFrame с операциями.
    car_ids : iterable, optional
        Список идентификаторов автомобилей для фильтрации.
    date_from : datetime-like, optional
        Начальная дата периода.
    date_to : datetime-like, optional
        Конечная дата периода.
    op_types : list, optional
        Список типов операций (например: ["payment", "fact", "debit"]).

    Возвращает
    ----------
    pandas.DataFrame
        Отфильтрованный DataFrame, содержащий только записи,
        удовлетворяющие заданным условиям.
    """
    res = df.copy()

    if car_ids is not None:
        res = res[res['car_id'].isin(car_ids)]

    if op_types is not None:
        res = res[res['type'].isin(op_types)]

    if date_from is not None:
        res = res[res['date'] >= date_from]

    if date_to is not None:
        res = res[res['date'] <= date_to]

    return res

def aggregate_ops(df, freq='7D'):
    """
    Агрегирует суммы операций по временным интервалам и типам операций.

    Функция группирует данные по дате с заданной частотой и типу операции,
    после чего возвращает сводную таблицу, удобную для визуализации
    (графики, диаграммы).

    Параметры
    ----------
    df : pandas.DataFrame
        DataFrame с колонками `date`, `type` и `amount`.
    freq : str, optional
        Частота агрегации:
        'D'  — день
        '7D' — 7 дней
        'W'  — календарная неделя
        'MS' — начало месяца
        'ME' — конец месяца
        'QE' — квартал
        'YE' — год

    Возвращает
    ----------
    pandas.DataFrame
        Pivot-таблицу, где:
        - индекс — дата
        - колонки — типы операций
        - значения — агрегированные суммы
    """
    weekly = (
        df
        .set_index('date')
        .groupby('type')['amount']
        .resample(freq)
        .sum()
        .reset_index()
    )

    pivot = (
        weekly
        .pivot(index='date', columns='type', values='amount')
        .fillna(0)
    )

    return pivot


# =========================
# Creating
# =========================

def make_payment_plan_df(
    car_ids,
    date_from,
    date_to,
    amount_by_car: dict,
    purchase_date_by_car: dict
):
    """
    Формирует плановые ежедневные платежи по автомобилям за заданный период.

    Для каждого автомобиля создаётся запись на каждый день расчётного периода
    с типом операции `payment_plan` и фиксированной суммой дохода. Если для
    автомобиля указана дата покупки, расчёт начинается не ранее этой даты.
    Автомобили, приобретённые после окончания периода, в расчёт не включаются.

    Параметры
    ----------
    car_ids : iterable
        Список идентификаторов автомобилей.
    date_from : datetime-like
        Начальная дата расчётного периода.
    date_to : datetime-like
        Конечная дата расчётного периода.
    amount_by_car : dict
        Словарь вида {car_id: сумма_в_день}, содержащий плановый доход
        для каждого автомобиля.
    purchase_date_by_car : dict
        Словарь вида {car_id: дата_покупки}, используемый для корректировки
        даты начала расчёта.

    Возвращает
    ----------
    pandas.DataFrame
        DataFrame с плановыми платежами, содержащий колонки:
        - date — дата планового платежа
        - car_id — идентификатор автомобиля
        - type — тип операции (`payment_plan`)
        - amount — плановая сумма дохода
    """

    rows = []

    for car_id in car_ids:
        purchase_date = purchase_date_by_car.get(car_id)

        # если дата покупки есть — сдвигаем старт
        start_date = (
            max(date_from, purchase_date)
            if pd.notna(purchase_date)
            else date_from
        )

        # если авто куплено ПОСЛЕ периода — пропускаем
        if start_date > date_to:
            continue

        dates = pd.date_range(start_date, date_to, freq='D')

        for d in dates:
            rows.append({
                'date': d,
                'car_id': car_id,
                'type': 'payment_plan',
                'amount': amount_by_car.get(car_id)
            })

    return pd.DataFrame(rows)

def build_avg_daily_summary_by_car(
    df,
    purchase_date_by_car,
    date_from,
    date_to
):
    """
    Строит сводную таблицу по автомобилям:
    - средний план в сутки
    - средний план по договорам в сутки
    - средний факт в сутки
    - отклонения в процентах от плана

    Возвращает DataFrame:
    car_id | plan_per_day | payment_per_day | fact_per_day |
    payment_deviation_pct | fact_deviation_pct
    """

    df = df.copy()

    # Оставляем только нужные типы
    df = df[df['type'].isin(['payment_plan', 'payment', 'fact'])]

    results = []

    for car_id, df_car in df.groupby('car_id'):

        purchase_date = purchase_date_by_car.get(car_id)
        if purchase_date is None:
            continue

        # Реальный период жизни авто
        real_start = max(date_from, purchase_date)
        days = (date_to - real_start).days + 1
        if days <= 0:
            continue

        # Суммы за период
        totals = (
            df_car
            .groupby('type')['amount']
            .sum()
            .to_dict()
        )

        plan_per_day = totals.get('payment_plan', 0) / days
        payment_per_day = totals.get('payment', 0) / days
        fact_per_day = totals.get('fact', 0) / days

        # Отклонения считаем только если есть план
        if plan_per_day > 0:
            payment_dev_pct = (payment_per_day - plan_per_day) / plan_per_day * 100
            fact_dev_pct = (fact_per_day - plan_per_day) / plan_per_day * 100
        else:
            payment_dev_pct = None
            fact_dev_pct = None

        results.append({
            'car_id': car_id,
            'План': int(plan_per_day),
            'По договору план': int(payment_per_day),
            'Факт': int(fact_per_day),
            'Отколнение договоров': payment_dev_pct,
            'Отклонение факта': fact_dev_pct
        })

    return pd.DataFrame(results).sort_values('car_id')

# =========================
# Validation
# =========================

def pct(part, total):
    """
    Не дает сломать программу в случае деления на 0.

    Параметры
    ----------
    part : float
        Принимает числитель
    total: float
        Принимает знаменатель
    Возвращает
    ----------
    число выражения: part/total * 100, если оно не делится на 0, в противном случае возвращает 0.
    """
    return (part / total * 100) if total else 0

# =========================
# Edition
# =========================
def scale_amount(df, column, factor):
    """
    Масштабирует денежные значения в целях анонимизации.
    Безопасно обрабатывает строки и пропущенные значения.
    Возвращает целочисленные значения.
    """
    df[column] = (
        pd.to_numeric(df[column], errors='raise')  # ← ключ
        .mul(factor)
        .round()
        .astype(int)
    )
    return df


def anonymize_car_id(df, column, car_map):
    """
    Заменяет исходные идентификаторы автомобилей анонимизированными с
    помощью словаря сопоставления.
    """
    mask = df[column].notna()

    df.loc[mask, column] = (
        df.loc[mask, column]
        .astype(str)
        .map(car_map)
        .fillna(df.loc[mask, column])
    )
    return df

def jitter_dates_in_range(
        df,
        column='date',
        max_days=5,
        min_date='2025-01-01',
        max_date='2025-12-31',
        seed=None
):
    """
    Случайным образом сдвигает даты на ±max_days.
    Сохраняет исходную дату, если сдвиг выходит за пределы [min_date, max_date].
    Сохраняет порядок событий.
    """
    if seed is not None:
        np.random.seed(seed)

    dates = pd.to_datetime(
        df[column],
        dayfirst=True,
        errors='coerce'
    )

    order = dates.sort_values().index
    base_dates = dates.loc[order]

    shifts = np.random.randint(-max_days, max_days + 1, size=len(base_dates))
    candidate = base_dates + pd.to_timedelta(shifts, unit='D')

    min_d = pd.to_datetime(min_date)
    max_d = pd.to_datetime(max_date)

    valid = (candidate >= min_d) & (candidate <= max_d)
    final_dates = candidate.where(valid, base_dates)

    final_dates = final_dates.cummax()

    df.loc[order, column] = final_dates
    return df

