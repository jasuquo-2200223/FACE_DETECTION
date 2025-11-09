from django import forms
from .models import UserUpload

class UploadForm(forms.ModelForm):
    class Meta:
        model = UserUpload
        fields = ['name', 'email', 'image']
