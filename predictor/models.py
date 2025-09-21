from django.db import models
from django.db import models
from django.utils import timezone

class StudentPrediction(models.Model):
    # Student data
    name = models.CharField(max_length=100)
    marks = models.FloatField(help_text="Marks out of 100")
    attendance = models.FloatField(help_text="Attendance percentage")
    study_hours = models.FloatField(default=0)
    previous_gpa = models.FloatField(default=0)
    assignments = models.FloatField(default=0)
    participation = models.FloatField(default=0)
    # ML prediction
    prediction = models.CharField(max_length=10, choices=[
        ('Pass', 'Pass'),
        ('Fail', 'Fail')
    ])
    prediction_probability = models.FloatField(help_text="Confidence score")
    
    # Metadata
    created_at = models.DateTimeField(default=timezone.now)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.name} - {self.prediction} ({self.prediction_probability:.2%})"
# Create your models here.
