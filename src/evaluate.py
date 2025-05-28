# src/evaluate.py

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


def report_and_plot(name, y_true, y_pred):
    # 1) Classification report
    print(f"\n{name} Classification Report")
    print(classification_report(y_true, y_pred, digits=4))

    # 2) Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"{name} Confusion Matrix")
    plt.colorbar()

    # Tick labels (0 = negative, 1 = positive)
    classes = ['neg', 'pos']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()
