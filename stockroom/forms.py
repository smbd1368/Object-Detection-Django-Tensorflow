from django import forms
from .models import TrainImage
import os

from django.conf import settings

train_folder = settings.TRAIN_FOLDER
class TrainDataUploadForm(forms.Form):
    def upload_train_data(self, train_folder):
        media_root = settings.MEDIA_ROOT
        for filename in os.listdir(train_folder):
            image_path = os.path.join('train/', filename)
            print(image_path)
            print(train_folder)
            print(filename)
            # Check if the image already exists in the TrainImage model
            if not TrainImage.objects.filter(image=image_path).exists():
                # Create a new TrainImage instance
                train_image = TrainImage(image=image_path, name=filename)
                train_image.save()
