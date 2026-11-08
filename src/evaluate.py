import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
from src.config import MODEL_SAVE_PATH

def evaluate_model():
    ensemble = joblib.load(MODEL_SAVE_PATH)
    X_te, y_test = joblib.load(MODEL_SAVE_PATH.parent / "test_data.pkl")
    
    print("\n================ EVALUATION METRICS ================")
    y_pred = ensemble.predict(X_te)
    y_prob = ensemble.predict_proba(X_te)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    
    print(f"Target Accuracy        : >= 95.00%")
    print(f"Achieved Test Accuracy : {acc * 100:.2f}%")
    print(f"ROC-AUC Score          : {roc:.4f}\n")
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Genuine (OR)", "Fake (CG)"], digits=4))
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["OR (Real)", "CG (Fake)"],
                yticklabels=["OR (Real)", "CG (Fake)"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix (Accuracy: {acc * 100:.2f}%)")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("Confusion matrix saved as 'confusion_matrix.png'")

if __name__ == "__main__":
    evaluate_model()