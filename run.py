import joblib
from src.train import train_ensemble
from src.evaluate import evaluate_model
from src.preprocess import normalize_text
from src.config import MODEL_SAVE_PATH, FEATURES_SAVE_PATH

def test_inference(review_text: str, category: str = "Home_and_Kitchen_5"):
    ensemble = joblib.load(MODEL_SAVE_PATH)
    pipeline = joblib.load(FEATURES_SAVE_PATH)
    
    clean = normalize_text(review_text)
    # Pass the category argument required by the updated feature pipeline
    features = pipeline.transform([review_text], [clean], [category])
    
    prediction = ensemble.predict(features)[0]
    prob = ensemble.predict_proba(features)[0]
    
    label = "FAKE / COMPUTER GENERATED (CG)" if prediction == 1 else "GENUINE / ORIGINAL (OR)"
    confidence = prob[1] if prediction == 1 else prob[0]
    
    print("\n--- Live Test Review ---")
    print(f"Category     : {category}")
    print(f"Input Review : {review_text}")
    print(f"Prediction   : {label} ({confidence * 100:.2f}% confidence)")

if __name__ == "__main__":
    print(">>> STEP 1: TRAINING ENSEMBLE PIPELINE...")
    train_ensemble()
    
    print("\n>>> STEP 2: RUNNING EVALUATION REPORT...")
    evaluate_model()
    
    print("\n>>> STEP 3: TESTING SAMPLE REVIEW...")
    sample_review = "This item is super wonderful! Best purchase ever made in my life!! WOW"
    test_inference(sample_review, category="Home_and_Kitchen_5")