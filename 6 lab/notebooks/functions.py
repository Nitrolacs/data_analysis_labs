import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from sklearn.decomposition import PCA


def visualization_with_pca(x, y, graphic_name, cluster_centers=None):
    plt.title(graphic_name)
    # Выполняем PCA на данных x
    pca = PCA(
        n_components=2,
        random_state=42)  # Выбираем количество компонент для снижения размерности
    x_pca = pca.fit_transform(x)  # Преобразуем данные в новое пространство

    # Строим диаграмму рассеяния по двум главным компонентам
    plt.scatter(x_pca[:, 0], x_pca[:, 1], c=y)  # Используем y для цвета точек

    if cluster_centers is not None:
        # Выполняем PCA на центрах кластеров
        cluster_centers_pca = pca.transform(
            cluster_centers)  # Преобразуем центры кластеров в новое пространство

        # Строим диаграмму рассеяния по двум главным компонентам для центров кластеров
        plt.scatter(cluster_centers_pca[:, 0], cluster_centers_pca[:, 1], c='red', marker='x',
                    s=200)  # Используем черный цвет и крестик для обозначения центров кластеров

    plt.xlabel(
        "Первая главная компонента")  # Подписываем ось x как первая главная компонента
    plt.ylabel(
        "Вторая главная компонента")  # Подписываем ось y как вторая главная компонента

    plt.show()  # Показываем диаграмму


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
    ax.set_ylim((0.2, 1.0))
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