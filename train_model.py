"""
ML Model Training Script for Student Pass/Fail Prediction

This script creates a dummy dataset and trains a simple ML model.
REPLACE THIS SECTION with your actual data science preprocessing/training code.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib

def generate_dummy_data(n_samples=1000):
    """
    Generate dummy student data for training.
    
    🔥 REPLACE THIS FUNCTION WITH YOUR ACTUAL DATA LOADING:
    - Load your CSV: df = pd.read_csv('your_data.csv')
    - Clean and preprocess your data
    - Feature engineering
    - Handle missing values, etc.
    """
    np.random.seed(42)
    
    # Generate synthetic student data
    marks = np.random.normal(65, 20, n_samples)  # Average 65, std 20
    attendance = np.random.normal(75, 15, n_samples)  # Average 75%, std 15%
    study_hours = np.random.normal(4, 2, n_samples)   # 0–12 hrs/day
    previous_gpa = np.random.normal(2.8, 0.8, n_samples)  # 0–4 GPA
    assignments = np.random.normal(80, 15, n_samples)  # %
    participation = np.random.normal(70, 20, n_samples) 
    # Clip values to realistic ranges
    marks = np.clip(marks, 0, 100)
    attendance = np.clip(attendance, 0, 100)
    study_hours = np.clip(study_hours, 0, 12)
    previous_gpa = np.clip(previous_gpa, 0, 4)
    assignments = np.clip(assignments, 0, 100)
    participation = np.clip(participation, 0, 100)

    # Create realistic pass/fail logic
    # Higher marks and attendance = higher chance of passing
    pass_probability = (
    (marks * 0.4) +              # marks: 40% weight
    (attendance * 0.2) +         # attendance: 20% weight
    (study_hours * 5 * 0.1) +    # study hours (scaled up, 0–12 → 0–60, 10% weight)
    (previous_gpa * 25 * 0.15) + (assignments * 0.1) +           # Assignment completion rate: 10% weight
    (participation * 0.05)          # Class participation: 5% weight
     ) / 100 
    pass_probability = np.clip(pass_probability, 0.1, 0.9)  # Add some randomness
    
    # Generate pass/fail labels
    outcome = np.random.binomial(1, pass_probability, n_samples)
    
    # Create DataFrame
    data = pd.DataFrame({
        'marks': marks,
        'attendance': attendance,
        'pass': outcome,
        'study_hours': study_hours,
    'previous_gpa': previous_gpa,
    'assignments': assignments,
    'participation': participation 
          # 1 = Pass, 0 = Fail
    })
    
    print("Sample data:")
    print(data.head(10))
    print(f"\nPass rate: {data['pass'].mean():.2%}")
    
    return data

def train_model():
    """
    Train the ML model.
    
    🔥 REPLACE/MODIFY THIS FUNCTION with your actual training logic:
    - Your feature engineering pipeline
    - Your model selection (RandomForest, XGBoost, Neural Network, etc.)
    - Your hyperparameter tuning
    - Your cross-validation strategy
    """
    print("🚀 Starting model training...")
    
    # Step 1: Load/generate data
    data = generate_dummy_data(1000)
    
    # Step 2: Prepare features and target
    X = data[['marks', 'attendance','study_hours',
    'previous_gpa',
    'assignments',
    'participation']]  # Features
    y = data['pass']  # Target (1 = Pass, 0 = Fail)
    
    # Step 3: Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Step 4: Create ML pipeline
    # 🔥 MODIFY THIS PIPELINE with your actual preprocessing and model:
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Normalize features
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5
        ))
    ])
    
    # Step 5: Train model
    print("Training model...")
    pipeline.fit(X_train, y_train)
    
    # Step 6: Evaluate model
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n✅ Model trained successfully!")
    print(f"Accuracy: {accuracy:.3f}")
    print("\nDetailed classification report:")
    print(classification_report(y_test, y_pred, target_names=['Fail', 'Pass']))
    
    # Step 7: Save model
    model_filename = 'student_model.pkl'
    joblib.dump(pipeline, model_filename)
    print(f"\n💾 Model saved as: {model_filename}")
    
    # # Step 8: Test model with sample predictions
    # print("\n🧪 Testing model with sample data:")
    # test_samples = [
    #     [85, 90],  # High marks, high attendance
    #     [45, 60],  # Low marks, moderate attendance
    #     [70, 85],  # Good marks, good attendance
    #     [30, 40],  # Low marks, low attendance
    # ]
    
    # for marks, attendance in test_samples:
    #     prediction = pipeline.predict([[marks, attendance]])[0]
    #     probability = pipeline.predict_proba([[marks, attendance]])[0]
    #     result = "Pass" if prediction == 1 else "Fail"
    #     confidence = max(probability)
        
    #     print(f"Marks: {marks}, Attendance: {attendance}% → {result} (confidence: {confidence:.3f})")

if __name__ == "__main__":
    train_model()
    print("\n🎉 Training complete! You can now run the Django app.")