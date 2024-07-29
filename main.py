import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import json

# Load the model
model_path = r'D:\Pycham\HWCR\handwritten_character_recognition_model.h5'  # Adjust the path accordingly
model = tf.keras.models.load_model(model_path)

# Load the class labels
with open(r'D:\Pycham\HWCR\class_labels.json', 'r') as f:  # Adjust the path accordingly
    class_labels = json.load(f)
    class_labels = {int(k): v for k, v in class_labels.items()}

def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((28, 28))  # Adjust size as per model input
    img = img.convert('L')  # Convert to grayscale if necessary
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    return img

def predict_character(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)[0]
    character = class_labels[predicted_class]
    return character

def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img = img.resize((200, 200))  # Resize for display purposes
        img = ImageTk.PhotoImage(img)
        panel.config(image=img)
        panel.image = img
        character = predict_character(file_path)
        result_label.config(text=f'Predicted Character: {character}')

# Create the main window
root = tk.Tk()
root.title("Character Recognition")

# Set window background color
root.configure(bg='#ADD8E6')

# Create a panel to display the image
panel = tk.Label(root, bg='#ADD8E6')
panel.pack(pady=10)

# Create a button to upload the image
upload_btn = tk.Button(root, text="Upload Image", command=upload_image, font=('Helvetica', 14), bg='#4CAF50', fg='white')
upload_btn.pack(pady=10)

# Create a label to display the result
result_label = tk.Label(root, text="Predicted Character: ", font=('Helvetica', 16, 'bold'), bg='#ADD8E6', fg='#FF4500')
result_label.pack(pady=20)

# Run the Tkinter event loop
root.mainloop()
