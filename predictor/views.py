from django.shortcuts import render, redirect
from django.contrib import messages
from django.http import JsonResponse
from django.urls import reverse
from .models import StudentPrediction
from .ml_model import predictor

def index(request):
    """Home page with student data entry form."""
    return render(request, 'index.html')

def predict_student(request):
    """Handle form submission and make ML prediction."""
    if request.method == 'POST':
        try:
            # Get form data
            name = request.POST.get('name', '').strip()
            marks = float(request.POST.get('marks', 0))
            attendance = float(request.POST.get('attendance', 0))
            study_hours = float(request.POST.get('study_hours', 0))
            previous_gpa = float(request.POST.get('previous_gpa', 0))
            assignments = float(request.POST.get('assignments', 0))
            participation = float(request.POST.get('participation', 0))
            # Validate input
            if not name:
                messages.error(request, "Please enter student name.")
                return redirect('predictor:index')
            
            if not (0 <= marks <= 100):
                messages.error(request, "Marks must be between 0 and 100.")
                return redirect('predictor:index')
            
            if not (0 <= attendance <= 100):
                messages.error(request, "Attendance must be between 0 and 100.")
                return redirect('predictor:index')
            
            # Make ML prediction
            result = predictor.predict(marks, attendance,study_hours,
    previous_gpa,
    assignments,
    participation)
            
            # Check if prediction was successful
            if not result.get('success', False):
                error_msg = result.get('error', 'ML model not available. Please run: python train_model.py')
                messages.error(request, f"Prediction failed: {error_msg}")
                return redirect('predictor:index')
            
            # Save to database
            student_prediction = StudentPrediction.objects.create(
                name=name,
                marks=marks,
                attendance=attendance,
                study_hours=study_hours,
                previous_gpa=previous_gpa,
                assignments=assignments,
                participation=participation,
                prediction=result['prediction'],
                prediction_probability=result['probability']
            )
            
            # Redirect to results page with student ID
            return redirect(f"{reverse('predictor:results')}?id={student_prediction.id}")
            
        except ValueError:
            messages.error(request, "Please enter valid numbers for marks and attendance.")
            return redirect('predictor:index')
        except Exception as e:
            messages.error(request, f"An error occurred: {str(e)}")
            return redirect('predictor:index')
    
    return redirect('predictor:index')

def results(request):
    """Display prediction results and recent submissions."""
    context = {}
    
    # Get specific result if ID provided
    student_id = request.GET.get('id')
    if student_id:
        try:
            context['current_result'] = StudentPrediction.objects.get(id=student_id)
        except StudentPrediction.DoesNotExist:
            messages.error(request, "Result not found.")
    
    # Get recent predictions for display
    context['recent_predictions'] = StudentPrediction.objects.all()[:10]
    
    # Check if model is loaded
    context['model_status'] = predictor.is_model_loaded()
    
    return render(request, 'results.html', context)