import matplotlib.pyplot as plt

# Константы для размеров фигуры и шрифта
FIG_SIZE = (16, 8)
FONT_SIZE = 18


def compare_regression_models(models_list: list, index: int,
                              param: str) -> None:
    """
    Построение графиков для сравнения моделей.
    :param models_list: Список кортежей с названиями моделей и значениями метрик
    :param index: Индекс значения метрики в кортеже
    :param param: название метрики
    :return: None
    """

    fig = plt.figure(figsize=FIG_SIZE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.bar(
        [model[0] for model in models_list],
        [model[index] for model in models_list],
    )
    ax.bar_label(
        ax.containers[0],
        label_type='edge',
        padding=3,
        fontsize=FONT_SIZE,
    )
    plt.ylabel('Значение', fontsize=FONT_SIZE)
    plt.title(f'Сравнение моделей по параметру {param}', fontsize=FONT_SIZE)
    plt.xticks(fontsize=FONT_SIZE - 4, rotation=90)
    plt.show()
