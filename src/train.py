import csv
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC

from src.config import DATA_PATH, MODEL_SAVE_PATH, FEATURES_SAVE_PATH, TEXT_COL, LABEL_COL, CAT_COL
from src.preprocess import normalize_text
from src.features import FeatureEngineeringPipeline

class FastEnsembleClassifier:
    def __init__(self, models, weights=[0.45, 0.35, 0.20]):
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
    raw_texts, clean_texts, categories, targets = [], [], [], []
    mapping = {'CG': 1, 'OR': 0}

    with open(DATA_PATH, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            lbl = row.get(LABEL_COL, '').strip()
            if lbl in mapping:
                txt = row.get(TEXT_COL, '')
                raw_texts.append(txt)
                clean_texts.append(normalize_text(txt))
                categories.append(row.get(CAT_COL, ''))
                targets.append(mapping[lbl])

    indices = np.arange(len(targets))
    y = np.array(targets)
    train_idx, test_idx, y_train, y_test = train_test_split(
        indices, y, test_size=0.15, random_state=42, stratify=y
    )

    print(f"Dataset split: {len(train_idx)} train samples, {len(test_idx)} test samples.")
    pipeline = FeatureEngineeringPipeline()
    pipeline.fit(
        [raw_texts[i] for i in train_idx],
        [clean_texts[i] for i in train_idx],
        [categories[i] for i in train_idx]
    )

    X_tr = pipeline.transform(
        [raw_texts[i] for i in train_idx],
        [clean_texts[i] for i in train_idx],
        [categories[i] for i in train_idx]
    )
    X_te = pipeline.transform(
        [raw_texts[i] for i in test_idx],
        [clean_texts[i] for i in test_idx],
        [categories[i] for i in test_idx]
    )

    print("Initializing ensemble models with category awareness...")
    models = [
        ('Calibrated LinearSVC', CalibratedClassifierCV(LinearSVC(C=1.0, dual=False, max_iter=3000, random_state=42), cv=3)),
        ('Logistic Regression', LogisticRegression(C=2.5, max_iter=1000, solver='lbfgs')),
        ('Calibrated SGD', SGDClassifier(loss='modified_huber', alpha=1e-5, max_iter=2000, random_state=42))
    ]

    ensemble = FastEnsembleClassifier(models=models, weights=[0.45, 0.35, 0.20])
    ensemble.fit(X_tr, y_train)

    joblib.dump(ensemble, MODEL_SAVE_PATH)
    joblib.dump(pipeline, FEATURES_SAVE_PATH)
    joblib.dump((X_te, y_test), MODEL_SAVE_PATH.parent / "test_data.pkl")
    print(f"Done! Models successfully saved to {MODEL_SAVE_PATH.parent}")

if __name__ == "__main__":
    train_ensemble()