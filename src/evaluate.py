import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from src.config import MODEL_SAVE_PATH

def evaluate_model():
    ensemble = joblib.load(MODEL_SAVE_PATH)
    X_te, y_test = joblib.load(MODEL_SAVE_PATH.parent / "test_data.pkl")

    y_pred = ensemble.predict(X_te)
    y_prob = ensemble.predict_proba(X_te)[:, 1]

    print("\n================ EVALUATION METRICS ================")
    print(f"Target Accuracy        : >= 95.00%")
    print(f"Achieved Test Accuracy : {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(f"ROC-AUC Score          : {roc_auc_score(y_test, y_prob):.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Genuine (OR)', 'Fake (CG)'], digits=4))

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    classes = ['Genuine (OR)', 'Fake (CG)']
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title='Confusion Matrix', ylabel='Actual', xlabel='Predicted')

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    print("Confusion matrix saved as 'confusion_matrix.png'")

if __name__ == "__main__":
    evaluate_model()