import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import MultipleLocator


def draw_graphics_bagg(parameters_values_tree: dict,
                       parameters_values_logreg: dict,
                       parameters_values_clas: dict,
                       title: str) -> None:
    """
    Отрисовка трех графиков с параметрами моделей.
    :param parameters_values_tree: Параметры дерева решений
    :param parameters_values_logreg: Параметры логистической регрессии
    :param parameters_values_clas: Параметры классификатора,
    :param title: Название ансамбля,
    :return: None
    """
    # Создаем список из трех словарей
    parameters_values_list = [parameters_values_tree, parameters_values_logreg,
                              parameters_values_clas]

    # Создаем рисунок с 3 подграфиками (1x3) и увеличиваем его размер
    gs = gridspec.GridSpec(1, 3)
    fig = plt.figure(figsize=(22, 6))

    # Перебираем список и индекс каждого словаря
    for i, parameters_values in enumerate(parameters_values_list):
        # Получаем координату подграфика из индекса
        x = i

        # Получаем название графика из словаря по ключу 'name'
        name = parameters_values['name']

        # Получаем значения параметра из словаря по ключу 'n_estimators'
        values = parameters_values['n_estimators']

        # Строим графики на соответствующем подграфике
        ax = fig.add_subplot(gs[x])  # row 0, col x
        ax.plot(values['range'], values['train_scores'],
                label='Тренировочная', color="blue")
        ax.set_xticks(list(values['range']))
        ax.plot(values['range'], values['test_scores'],
                label='Тестовая', color="red")
        ax.set_xlabel('n_estimators')  # Только целочисленные значения
        ax.set_ylabel('ROC_AUC')
        ax.legend()
        ax.set_title(name)  # Устанавливаем название графика

        # Делаем график квадратным и равномерным по сторонам
        # ax.set_aspect('equal')

    # Делаем больше расстояния между графиками
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    fig.suptitle(title)

    plt.show()


def draw_graphics_boost(parameters_values: dict, title: str) -> None:
    """
    Отрисовка двух графиков с параметрами моделей
    :param parameters_values: Параметры
    :param title: Заголовок
    :return: None
    """
    # Создаем рисунок с 2 подграфиками (1x2) и увеличиваем его размер
    gs = gridspec.GridSpec(1, 2)
    fig = plt.figure(figsize=(18, 8))

    # Получаем название графиков из словаря по ключу 'name'
    name = parameters_values['name']

    # Перебираем ключи словаря, кроме 'name'
    for i, key in enumerate(parameters_values):
        if key == 'name':
            continue

        # Получаем координату подграфика из индекса
        x = i

        # Получаем значения параметра из словаря по ключу
        values = parameters_values[key]

        # Строим графики на соответствующем подграфике
        ax = fig.add_subplot(gs[x])  # row 0, col x
        ax.plot(values['range'], values['train_scores'],
                label='Тренировочная', color="blue")
        ax.set_xticks(list(values['range']))
        ax.plot(values['range'], values['test_scores'],
                label='Тестовая', color="red")
        ax.set_xlabel(key)  # Только целочисленные значения
        ax.set_ylabel('ROC_AUC')
        ax.legend()

        ax.set_title(name)  # Устанавливаем название графика

    # Делаем больше расстояния между графиками
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    fig.suptitle(title)

    plt.show()


def draw_graphics_stack(results: list, model_name: str, score: str) -> None:
    """
    Отрисовка графика для стекинга
    :param results: Список с результатами
    :param model_name: Название модели
    :param score: Оценка
    :return: None
    """
    fig, ax = plt.subplots()
    fig.suptitle(model_name)

    comb_names = ["\n".join(names) for names, _, _ in results]

    train_scores = [score for _, score, _ in results]

    test_scores = [score for _, _, score in results]

    ax.bar(comb_names, train_scores, color="blue", label="Тренировочная")
    ax.bar(comb_names, test_scores, color="red", label="Тестовая")
    ax.set_xticks(range(len(comb_names)))
    ax.set_xticklabels(comb_names)

    ax.set_xlabel("Комбинации базовых алгоритмов")

    ax.set_ylabel(score)
    ax.set_ylim([min(test_scores) - 0.01, max(train_scores) + 0.01])

    ax.legend()
    plt.tight_layout()
    plt.show()

    index = test_scores.index(max(test_scores))
    print(
        f"Лучшая комбинация:\n{comb_names[index]}\nОценка {score} на тестовой "
        f"выборке:\n{round(test_scores[index], 3)}")


def plot_histogram(data: dict, score: str, name: str) -> None:
    """
    Построение гистограммы для сравнения различных методов.
    :param data: Словарь с данными для сравнения
    :param score: Оценка
    :param name: Заголовок графика
    :return: None
    """

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
    ax.set_title(name, y=1.2)

    # добавляем значение над каждым столбцом
    for rect, score in zip(ax.patches, roc_auc_values):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.01,
                score, ha="center")

    # показываем график
    plt.show()
