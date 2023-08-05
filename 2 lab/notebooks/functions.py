from sklearn.metrics import roc_curve, auc
from matplotlib.ticker import MultipleLocator


def plot_roc_curves(ax, probs, names, y_test, axes):
    """
    Строит ROC кривые для всех моделей
    :param ax: ось, на которой строим график
    """
    labels = tuple()
    for index, prob in enumerate(probs):
        fpr, tpr, _ = roc_curve(y_test, prob)
        area = round(auc(fpr, tpr), 3)
        labels += (names[index] + f" ({area=})",)
        axes[0].plot(fpr, tpr)
    ax.plot((-0.01, 1.01), (-0.01, 1.01), color="navy", linestyle="--")
    ax.set_title("ROC curves")
    ax.set_xlim((-0.01, 1.01))
    ax.set_ylim((-0.01, 1.01))
    ax.set_xlabel("False positive rate", labelpad=15)
    ax.set_ylabel("True positive rate", labelpad=15)
    ax.legend(loc="lower right", labels=labels)


def plot_bar_graph(ax, names, scoring):
    """
    Строит столбчатую диаграмму для score моделей
    :param ax: ось, на которой строим график
    """
    ax.bar(names, scoring)
    ax.set_title("Scoring (ROC-AUC)")
    ax.tick_params("x", labelrotation=90)
    ax.set_ylim((0.5, 1))
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(MultipleLocator(0.01))
    for rect, score in zip(ax.patches, scoring):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.01,
                score, ha="center")
    ax.set_xlabel("Model", labelpad=15)
    ax.set_ylabel("Score", labelpad=15)
