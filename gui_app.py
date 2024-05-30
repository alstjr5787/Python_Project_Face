import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2
from mtcnn import MTCNN
from keras.models import load_model

face_recognition_model = load_model('face_recognition_model.h5')
mtcnn_detector = MTCNN()

def detect_faces(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    faces = mtcnn_detector.detect_faces(image_rgb)

    for face in faces:
        x, y, w, h = face['box']
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)


        face_img = image[y:y+h, x:x+w]
        face_img = cv2.GaussianBlur(face_img, (99, 99), 30)
        image[y:y+h, x:x+w] = face_img

        face_img = cv2.resize(face_img, (100, 100))
        face_img = np.expand_dims(face_img, axis=0)

        prediction = face_recognition_model.predict(face_img)
        predicted_class = np.argmax(prediction)
        if predicted_class == 0:
            label_text = "Person"
            label_color = (0, 255, 0)
            label_confidence = "True"
        else:
            label_text = "Non-Person"
            label_color = (255, 255, 255)
            label_confidence = "False"

        confidence = np.max(prediction)
        text = f"{label_text} ({confidence:.0%})"
        text_width, text_height = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.putText(image, label_text, (x + (w - text_width) // 2, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 2)
        cv2.putText(image, f"{label_confidence} ({confidence:.0%})", (x + (w - text_width) // 2, y - 10 + text_height + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image.thumbnail((300, 300))
    photo = ImageTk.PhotoImage(image)

    canvas_width = canvas.winfo_width()
    canvas_height = canvas.winfo_height()
    x_center = (canvas_width - photo.width()) / 2
    y_center = (canvas_height - photo.height()) / 2

    canvas.create_image(x_center, y_center, anchor='nw', image=photo)
    canvas.image = photo

    num_faces = len(faces)
    if num_faces > 0:
        label.config(text=f"{num_faces}명의 얼굴이 감지되었습니다.")
    else:
        label.config(text="얼굴이 감지되지 않았습니다.")

def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.gif")])
    if file_path:
        detect_faces(file_path)

root = tk.Tk()
root.title("Face mosaic")

canvas = tk.Canvas(root, width=300, height=300)
canvas.pack()

upload_button = tk.Button(root, text="이미지 업로드", command=upload_image)
upload_button.pack()

label = tk.Label(root, text="")
label.pack()

root.mainloop()
