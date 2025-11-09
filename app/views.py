from django.shortcuts import render
from .forms import UploadForm
from .models import UserUpload
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from django.core.files.storage import default_storage

# Load your model once when the server starts
model = load_model('face_emotionModel.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def index(request):
    prediction_text = None
    image_url = None

    if request.method == 'POST':
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            upload = form.save()

            # Save image to disk to get proper path
            image_file = upload.image
            image_path = default_storage.path(image_file.name)

            # Read image properly with OpenCV
            img = cv2.imdecode(
                np.frombuffer(image_file.read(), np.uint8),
                cv2.IMREAD_COLOR
            )

            # Safety check
            if img is None:
                prediction_text = "Error: Could not read image. Please try another image."
            else:
                # Convert to gray for face detection
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                if len(faces) == 0:
                    prediction_text = "No face detected in the image."
                else:
                    for (x, y, w, h) in faces:
                        face = img[y:y+h, x:x+w]
                        face = cv2.resize(face, (48, 48))
                        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                        face = face / 255.0
                        face = np.reshape(face, (1, 48, 48, 1))

                        prediction = model.predict(face)
                        emotion_index = int(np.argmax(prediction))
                        emotion = emotion_labels[emotion_index]
                        prediction_text = f"You look {emotion}. Why are you {emotion.lower()}?"
                        break

            image_url = upload.image.url
    else:
        form = UploadForm()

    return render(request, 'index.html', {'form': form, 'prediction_text': prediction_text, 'image_url': image_url})
