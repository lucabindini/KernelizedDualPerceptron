import numpy as np
import matplotlib.pyplot as plt
from random import random


def accuracies_kernels_bar_plot(kernel1, kernel2, kernel3, ker1_accuracies, ker2_accuracies, ker3_accuracies):
    ind = np.arange(3)
    width = 0.30
    fig, ax = plt.subplots()

    rects1 = ax.bar(ind, ker1_accuracies, width, color='blue')
    rects2 = ax.bar(ind + width, ker2_accuracies, width, color='yellow')
    rects3 = ax.bar(ind + 2 * width, ker3_accuracies, width, color='green')

    ax.set_xlabel('Datasets')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy per Kernel')
    ax.set_xticks(ind + 2 * width / 2)
    ax.set_xticklabels((kernel1, kernel2, kernel3))
    ax.legend((rects1[0], rects2[0], rects3[0]), ('linear', 'polynomial', 'RBF'))
    def autolabel(rects):
        for rect in rects:
            height = float(rect.get_height())
            ax.text(rect.get_x() + rect.get_width() / 2, height, height, ha='center')
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.savefig("img/accuracies.png", dpi=300, format='png', bbox_inches='tight')
    plt.show()
    plt.clf()


def validation_error_rate_plot(data):
    x = np.arange(1, len(data) + 1, 1)
    y = data
    plt.plot(x, y, '-ro')
    plt.title('Validation Error Rate')
    plt.xlabel('Epochs')
    plt.ylabel('Error rate')
    plt.savefig("img/" + str(random()) + ".png", dpi=300, format='png', bbox_inches='tight')
    plt.clf()
