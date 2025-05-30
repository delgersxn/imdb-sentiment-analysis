from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def report_and_plot(name, y_true, y_pred):
    # 1) Classification report
    print(f"\n{name} Classification Report")
    print(classification_report(y_true, y_pred, digits=4))

    # 2) Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # print(cm) 
    classes = ['neg', 'pos']

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()

    filename = name.replace(" ", "_").lower()
    plt.savefig(f"confusion_matrix/{filename}.png")
    plt.show(block=False)