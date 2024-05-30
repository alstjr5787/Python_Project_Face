import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

dataset_dir = "./faces1"

class_labels = os.listdir(dataset_dir)

data = []
labels = []

for label, class_name in enumerate(class_labels):
    class_dir = os.path.join(dataset_dir, class_name)
    if os.path.isdir(class_dir):
        for image_name in os.listdir(class_dir):
            if image_name.endswith('.jpg') or image_name.endswith('.png'):
                image_path = os.path.join(class_dir, image_name)
                image = cv2.imread(image_path)
                image = cv2.resize(image, (100, 100))
                data.append(image)
                labels.append(label)


data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(class_labels), activation='softmax')
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(trainX, trainY, epochs=1000, validation_data=(testX, testY))

model.save("face_recognition_model.h5")
