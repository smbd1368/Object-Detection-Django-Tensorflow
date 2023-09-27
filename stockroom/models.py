from django.db import models

from django.conf import settings
train_folder = settings.TRAIN_FOLDER
import os

class TrainImage(models.Model):
    image = models.ImageField(upload_to='train/', null=True, blank=True)
    name = models.CharField(max_length=100,null=True, blank=True)
    desc = models.TextField(null=True, blank=True)

    def __str__(self):
        return self.name
    
    