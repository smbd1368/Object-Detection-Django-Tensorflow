from django.shortcuts import render, redirect
from .forms import TrainDataUploadForm
from .models import TrainImage
from django.conf import settings
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
        
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import numpy as np


def upload_train_data(request):
    if request.method == 'POST':
        form = TrainDataUploadForm(request.POST)

        if form.is_valid():
            # train_folder = '/home/bagher/Downloads/image_processing/train'
            form.upload_train_data(settings.TRAIN_FOLDER)

            return redirect('success')
    else:
        form = TrainDataUploadForm()

    return render(request, 'stockroom/upload_form.html', {'form': form})

def upload_success(request):
    return render(request, 'stockroom/upload_success.html')



from .models import TrainImage
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity as CS
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage

# Define the LowResourceCNN model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


def preprocess_image(img):
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

import io

def upload_form(request):
    if request.method == 'POST':
        test_image = request.FILES['test_image']
        fs = FileSystemStorage()
        filename = fs.save(test_image.name, test_image)
        uploaded_file_url = fs.url(filename)
        print(uploaded_file_url)
        print(filename)

        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(10, activation='softmax'))
        # model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
        # model.fit(train_data, train_labels, epochs=10, batch_size=32)
        
        # model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

        # Load the pre-trained VGG16 model
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

        # Remove the top classification layers
        feature_extractor = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)


        train_images = TrainImage.objects.all()
        train_image_features = []
        train_image_paths = []

        for train_image in train_images:
            img = image.load_img(train_image.image.path, target_size=(224, 224))
            img = preprocess_image(img)
            features = feature_extractor.predict(img)
            train_image_features.append(features.flatten())
            train_image_paths.append(train_image.image.url)

        train_image_features = np.array(train_image_features)

        img = image.load_img('media/' + filename, target_size=(224, 224))
        img = preprocess_image(img)
        test_image_features = model.predict(img)

        # similarities = CS(test_image_features.reshape(1, -1), train_image_features)
        
        from sklearn.decomposition import PCA
        pca = PCA(n_components=10)
        train_features_pca = pca.fit_transform(train_image_features)

        # Calculate cosine similarity
        similarities = CS(test_image_features, train_features_pca)


        print(similarities,"CDCCCC((((((((((((((((((()))))))))))))))))))")
        most_similar_indices = np.argsort(similarities)[::-1]
        print(most_similar_indices,"DCCDCDCDCDCDCDCDCDCD")

        similar_images = []
        for index in most_similar_indices[0]:
            similar_image_path = train_image_paths[index]
            similar_images.append(similar_image_path)
        print(similar_images,"!!!!!!!!!!!!!!!!!!!!!")
        print(train_image_paths,"@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        similar_images = [similar_images[-1]]
        print(similar_images)
        train_imagess = TrainImage.objects.filter(image=similar_images[0].replace("/media/",""))
        print(train_imagess)
        s =TrainImage.objects.get(image=similar_images[0].replace("/media/",""))
        similar_images=[s.image.url]
        return render(request, 'stockroom/result.html', 
                      {'uploaded_file_url': uploaded_file_url, 
                       'similar_images': similar_images})

    return render(request, 'stockroom/upload_file.html')



