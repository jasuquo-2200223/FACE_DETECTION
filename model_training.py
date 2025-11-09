# model_training.py
# ----------------------------
# This script builds and saves a simple CNN model
# for facial emotion detection.
# (You can later train it properly using FER2013 dataset.)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np

# ----------------------------
# 1️⃣ Define a simple CNN model
# ----------------------------
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # 7 emotions: angry, disgust, fear, happy, sad, surprise, neutral
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ----------------------------
# 2️⃣ Fake training data (just to create and save the model quickly)
# ----------------------------
X_fake = np.random.rand(10, 48, 48, 1)
y_fake = tf.keras.utils.to_categorical(np.random.randint(7, size=(10, 1)), num_classes=7)

model.fit(X_fake, y_fake, epochs=1, verbose=1)

# ----------------------------
# 3️⃣ Save the model
# ----------------------------
model.save('face_emotionModel.h5')

print("✅ Model saved successfully as 'face_emotionModel.h5'")
