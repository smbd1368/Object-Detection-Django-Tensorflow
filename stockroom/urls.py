from django.urls import path
from . import views

from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path('upload/', views.upload_train_data, name='upload_train_data'),
    path('success/', views.upload_success, name='success'),
    path('detect/', views.upload_form, name='upload_form'),
]

#urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)


