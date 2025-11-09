from django.contrib import admin
from .models import UserUpload  # <- use the correct model name

admin.site.register(UserUpload)

