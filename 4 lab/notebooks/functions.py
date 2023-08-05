import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


def draw_graphics(parameters_values: dict, score: str):
    """
    Отрисовка четырех графиков с параметрами дерева.
    :param parameters_values: Параметры дерева решений
    :param score: Оценка
    :return: None
    """

    # Создаем рисунок с 4 подграфиками (2x2)
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))

    # Получаем список ключей словаря
    keys = list(parameters_values.keys())

    # Перебираем индексы и ключи с помощью функции enumerate()
    for i, key in enumerate(keys):

        # Получаем координаты подграфика из индекса
        x = i // 2
        y = i % 2

        # Получаем значения параметра из словаря по ключу
        values = parameters_values[key]

        # Строим графики на соответствующем подграфике
        axes[x, y].plot(values['range'], values['train_scores'],
                        label='Тренировочная', color="blue")
        axes[x, y].set_xticks(list(values['range']))
        axes[x, y].plot(values['range'], values['test_scores'],
                        label='Тестовая', color="red")
        axes[x, y].set_xlabel(key)  # Только целочисленные значения
        axes[x, y].set_ylabel(score)
        axes[x, y].legend()
    plt.show()


# определяем функцию для построения гистограммы
def plot_histogram(data, score):

    # проверяем, что data - это словарь
    if not isinstance(data, dict):
        print("Неверный формат данных. Нужен словарь.")
        return

    # получаем список названий алгоритмов и список значений roc_auc
    algorithms = list(data.keys())
    roc_auc_values = list(data.values())

    # создаем фигуру и оси для графика
    fig, ax = plt.subplots()

    # строим гистограмму с двумя столбцами и увеличиваем расстояние между ними
    ax.bar(algorithms, roc_auc_values)

    ax.tick_params("x", labelrotation=90)
    ax.set_ylim((0.5, 1))
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(MultipleLocator(0.01))

    # добавляем подписи к осям и заголовок
    ax.set_xlabel("Название алгоритма", labelpad=15)
    ax.set_ylabel(f"Значение {score}", labelpad=15)
    ax.set_title(f"Сравнение алгоритмов по {score}", y=1.2)

    # добавляем значение над каждым столбцом
    for rect, score in zip(ax.patches, roc_auc_values):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.01,
                score, ha="center")

    # показываем график
    plt.show()
