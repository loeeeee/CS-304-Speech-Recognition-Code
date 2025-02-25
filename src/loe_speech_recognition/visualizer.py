
import numpy as np
import matplotlib.pyplot as plt


def plot_confusion_matrix_from_lists(predictions, ground_truth, class_names, title='Confusion Matrix', figsize=(8, 6)):
    """
    Plots a confusion matrix from lists of predictions and ground truth.

    Args:
        predictions (list): List of predicted class labels.
        ground_truth (list): List of true class labels.
        class_names (list): List of class names (labels) for the axes.
        title (str, optional): Title of the plot. Defaults to 'Confusion Matrix'.
        cmap (matplotlib.colors.Colormap, optional): Matplotlib colormap. Defaults to plt.cm.Blues.
        figsize (tuple, optional): Figure size (width, height). Defaults to (8, 6).
    """

    # Create the confusion matrix
    num_classes = len(class_names)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

    for true_label, predicted_label in zip(ground_truth, predictions):
        true_index = class_names.index(true_label)
        predicted_index = class_names.index(predicted_label)
        confusion_matrix[true_index, predicted_index] += 1

    plt.figure(figsize=figsize)
    plt.imshow(confusion_matrix, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = confusion_matrix.max() / 2.
    for i, j in np.ndindex(confusion_matrix.shape):
        plt.text(j, i, format(confusion_matrix[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if confusion_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f"./plots/confusion_matrix_{title}.png")
