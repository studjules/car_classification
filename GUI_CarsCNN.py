import tkinter as tk
from tkinterdnd2 import TkinterDnD, DND_FILES
from PIL import Image, ImageTk
import torch
import torchvision.transforms as transforms
from CarsCNN_class import CarsCNN_final
import socket

car_classes = ['Golf', 'bmw serie 1', 'chevrolet spark',
               'chevroulet aveo', 'clio', 'duster', 'hyundai i10',
               'hyundai tucson', 'logan', 'megane', 'mercedes class a',
               'nemo citroen', 'octavia', 'picanto', 'polo', 'sandero',
               'seat ibiza', 'symbol', 'toyota corolla', 'volkswagen tiguan']
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(('0.0.0.0', 9999))

class CarClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Car Image Classifier")

        self.model = self.load_model()

        self.create_widgets()

    def create_widgets(self):
        self.label = tk.Label(self.root, text="Drag and drop an image here:", font=("Helvetica", 22))
        self.label.pack(pady=10)

        self.image_label = tk.Label(self.root)
        self.image_label.pack()

        # Enable drop functionality
        self.label.drop_target_register(DND_FILES)
        self.label.dnd_bind('<<Drop>>', self.load_image)

        self.classify_button = tk.Button(self.root, text="Classify", command=self.classify_image, font=("Helvetica", 22))
        self.classify_button.pack(pady=10)

        self.delete_button = tk.Button(self.root, text="Delete Image", command=self.delete_image, font=("Helvetica", 22))
        self.delete_button.pack(pady=10)

        self.result_label = tk.Label(self.root, text="", font=("Helvetica", 14, "bold"))
        self.result_label.pack(pady=10)

    def load_model(self):
        # Load the saved model
        checkpoint = torch.load("car_classification_model.pth")
        model = CarsCNN_final()
        model.load_state_dict(checkpoint)
        model.eval()
        return model

    def load_image(self, event):
        file_path = event.data
        self.image = Image.open(file_path)
        self.image.thumbnail((300, 300))
        self.photo = ImageTk.PhotoImage(self.image)
        self.image_label.config(image=self.photo)

    def delete_image(self):
        # Clear the image and display default text
        self.image_label.config(image="")
        self.label.config(text="Drag and drop an image here:")
        self.result_label.config(text="")

    def classify_image(self):
        if hasattr(self, 'image'):
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
            image_tensor = transform(self.image).unsqueeze(0)

            with torch.no_grad():
                output = self.model(image_tensor)
                _, predicted = torch.max(output.data, 1)
                predicted_class = car_classes[predicted.item()]

            self.result_label.config(text=f"Predicted Class: {predicted_class}")
        else:
            self.result_label.config(text="Please drop an image or use the 'Classify' button.")

if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = CarClassifierGUI(root)
    root.geometry("600x500")
    root.mainloop()

