from django.contrib import admin
from .models import StudentPrediction
@admin.register(StudentPrediction)
class StudentPredictionAdmin(admin.ModelAdmin):
    list_display=('name',"marks","attendance","prediction","prediction_probability","created_at")
    list_filter=("prediction",'created_at')
    search_fields=('name',)
    readonly_fields=('created_at',)
    ordering=("-created_at",)
    fieldsets=(
        ('Student Information',{'fields':('name',"marks","attendance")}),
        ("ML Prediction",{'fields':('prediction',"prediction_probability")}),
        ("Metadata",{
            'fields':("created_at",),
            "classes":("collapse",)
        })
    )

# Register your models here.
