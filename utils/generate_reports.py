import numpy as np
import os
from sklearn.metrics import classification_report, balanced_accuracy_score


def generate_classification_report(model, test_ds, class_names, model_name, img_size):
    y_true, y_pred = [], []

    for images, labels in test_ds:
        preds = model.predict(images)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    print(f'Balanced Accuracy: \t{balanced_acc:.4f}')

    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)


    report_path = os.path.join(f"classification_report_{balanced_acc:.3f}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f'Classification Report\n\n')
        f.write(f'Balanced Accuracy: {balanced_acc:.3f}\n\n')
        f.write(report)
        f.write("\n\nModel Summary:\n")
        f.write(f"Model: {model_name} for size {img_size}x{img_size}\n")
        model.summary(print_fn=lambda x: f.write(x + "\n"))

    print(f"Report saved in: {report_path}")
    return balanced_acc
