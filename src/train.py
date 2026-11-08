import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from lightgbm import LGBMClassifier

from src.config import DATA_PATH, MODEL_SAVE_PATH, FEATURES_SAVE_PATH, TEXT_COL, LABEL_COL, CAT_COL
from src.preprocess import normalize_text, parse_labels
from src.features import FeatureEngineeringPipeline

class FastEnsembleClassifier:
    def __init__(self, models, weights=[0.40, 0.35, 0.25]):
        self.models = models
        self.weights = weights

    def fit(self, X, y):
        for name, model in self.models:
            print(f"  Fitting {name}...")
            model.fit(X, y)
        return self

    def predict_proba(self, X):
        probs = [model.predict_proba(X) for _, model in self.models]
        return sum(w * p for w, p in zip(self.weights, probs))

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)

def train_ensemble():
    print(f"Loading dataset from: {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    
    print("Normalizing text and mapping labels (CG -> 1, OR -> 0)...")
    df['clean_text'] = df[TEXT_COL].apply(normalize_text)
    df['target'] = parse_labels(df[LABEL_COL])
    df = df.dropna(subset=['target'])
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df[[TEXT_COL, 'clean_text', CAT_COL]],
        df['target'],
        test_size=0.15,
        random_state=42,
        stratify=df['target']
    )
    
    print(f"Dataset split: {len(X_train_raw)} train samples, {len(X_test_raw)} test samples.")
    print("Building category, stylometric & n-gram feature matrices...")
    
    pipeline = FeatureEngineeringPipeline()
    pipeline.fit(X_train_raw[TEXT_COL], X_train_raw['clean_text'], X_train_raw[CAT_COL])
    
    X_tr = pipeline.transform(X_train_raw[TEXT_COL], X_train_raw['clean_text'], X_train_raw[CAT_COL])
    X_te = pipeline.transform(X_test_raw[TEXT_COL], X_test_raw['clean_text'], X_test_raw[CAT_COL])
    
    print("Initializing ensemble models with category awareness...")
    models = [
        ('Calibrated LinearSVC', CalibratedClassifierCV(LinearSVC(C=1.0, dual=False, max_iter=3000, random_state=42), cv=3)),
        ('Logistic Regression', LogisticRegression(C=2.5, max_iter=1000, solver='lbfgs')),
        ('LightGBM', LGBMClassifier(n_estimators=200, learning_rate=0.08, num_leaves=31, n_jobs=-1, random_state=42, verbose=-1))
    ]
    
    ensemble = FastEnsembleClassifier(models=models, weights=[0.40, 0.35, 0.25])
    ensemble.fit(X_tr, y_train)
    
    joblib.dump(ensemble, MODEL_SAVE_PATH)
    joblib.dump(pipeline, FEATURES_SAVE_PATH)
    joblib.dump((X_te, y_test), MODEL_SAVE_PATH.parent / "test_data.pkl")
    print(f"Done! Models successfully saved to {MODEL_SAVE_PATH.parent}")

if __name__ == "__main__":
    train_ensemble()