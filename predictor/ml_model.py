"""
ML Model Integration for Django

This module handles loading the trained ML model and making predictions.
"""

import joblib
import numpy as np
import os
from django.conf import settings

class StudentPredictor:
    def __init__(self):
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the trained ML model from file."""
        try:
            # Look for model in root directory
            model_path = os.path.join(settings.BASE_DIR, 'student_model.pkl')
            if os.path.exists(model_path):
                loaded_data = joblib.load(model_path)
                
                # Check if it's the enhanced format with metadata
                if isinstance(loaded_data, dict) and 'model' in loaded_data:
                    self.model = loaded_data['model']
                    print(f"✅ Enhanced ML model loaded successfully from {model_path}")
                    print(f"   Training accuracy: {loaded_data.get('training_accuracy', 'N/A')}")
                    print(f"   Features: {loaded_data.get('feature_columns', ['marks', 'attendance'])}")
                else:
                    # Simple model format (sklearn pipeline directly)
                    self.model = loaded_data
                    print(f"✅ Simple ML model loaded successfully from {model_path}")
                    
            else:
                print(f"❌ Model file not found at {model_path}")
                print("Please run 'python train_model.py' first to generate the model.")
                self.model = None
        except Exception as e:
            print(f"❌ Error loading ML model: {e}")
            self.model = None
    
    def predict(self, marks, attendance, study_hours, previous_gpa, assignments, participation):
        """
        Make prediction for a student using all features.
        
        Args:
            marks (float): Student marks (0-100)
            attendance (float): Student attendance percentage (0-100)
            study_hours (float): Study hours per day (0-12)
            previous_gpa (float): Previous GPA (0-4)
            assignments (float): Assignment completion rate (%) (0-100)
            participation (float): Class participation (%) (0-100)
        
        Returns:
            dict: {
                'prediction': 'Pass' or 'Fail',
                'probability': float (confidence score),
                'success': bool,
                'error': str (if success is False)
            }
        """
        if self.model is None:
            return {
                'prediction': 'Unknown',
                'probability': 0.0,
                'success': False,
                'error': 'Model not loaded. Please run: python train_model.py'
            }
        
        try:
            # Prepare input data with proper feature names
            import pandas as pd
            input_data = pd.DataFrame([[
                marks, attendance, study_hours, previous_gpa, assignments, participation
            ]], columns=[
                'marks', 'attendance', 'study_hours', 'previous_gpa', 'assignments', 'participation'
            ])
            
            # Make prediction
            prediction_raw = self.model.predict(input_data)[0]
            prediction_proba = self.model.predict_proba(input_data)[0]
            
            # Convert to human-readable format
            prediction = "Pass" if prediction_raw == 1 else "Fail"
            confidence = max(prediction_proba)  # Highest probability
            
            return {
                'prediction': prediction,
                'probability': confidence,
                'success': True
            }
            
        except Exception as e:
            return {
                'prediction': 'Unknown',
                'probability': 0.0,
                'success': False,
                'error': f'Prediction error: {str(e)}'
            }

    def is_model_loaded(self):
     """Check if model is properly loaded."""
     return self.model is not None

# Global predictor instance

predictor = StudentPredictor()
