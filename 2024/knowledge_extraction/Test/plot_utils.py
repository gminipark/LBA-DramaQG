import matplotlib.pyplot as plt
import numpy as np

def plot_results(data, indices, title, labels, line=True, scatter=False, bar=False):
    num_results = len(data)
    for i in range(num_results):
        unique_values, counts = np.unique(data[i], return_counts=True)
        if line:
            # plt.plot(indices, data[i], linestyle='-', linewidth=1, label=f'{labels[i]} (Line)')
            plt.plot(unique_values, counts, linestyle='-', linewidth=3, label=f'{labels[i]} (Line)')
        if scatter:
            # plt.scatter(indices, data[i], s=50, label=f'{labels[i]} (Dots)')
            plt.scatter(unique_values, counts, s=150, label=f'{labels[i]} (Dots)')
        if bar:
            width = 0.015
            x_shifted = unique_values + i*(width)
            plt.bar(x_shifted, counts, width=width, alpha=0.7, label=f'{labels[i]}')

    plt.xlabel('Achieved Accuracy Scores')
    plt.ylabel('Counts')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.xlim(left=-0.02)  # X-axis starts from 0
    plt.ylim(bottom=-1)  # Y-axis starts from -1


def errorCase_analysis(gold_list, extracted_list):
    """
    gold_list (list of list): Gold Extraction 내용이 케이스별로 리스트에 담겨있음
    extracted_list (list of list): 실제 Extract한 내용이 케이스별로 리스트에 담겨있음
    """
    assert len(gold_list) == len(extracted_list)
    