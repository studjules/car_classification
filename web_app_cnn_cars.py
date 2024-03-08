from flask import Flask, render_template, request
import torch
import torchvision.transforms as transforms
from PIL import Image
from CarsCNN_class import CarsCNN_final
from torchvision.datasets import ImageFolder

car_classes = ['Golf', 'bmw serie 1', 'chevrolet spark',
           'chevroulet aveo', 'clio', 'duster', 'hyundai i10',
           'hyundai tucson', 'logan', 'megane', 'mercedes class a',
           'nemo citroen', 'octavia', 'picanto', 'polo', 'sandero',
           'seat ibiza', 'symbol', 'toyota corolla', 'volkswagen tiguan']
app = Flask(__name__)
# Define the root directory where your dataset is located
root_directory = "DATA"

# Define the transformation to be applied to the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Adjust as needed
    transforms.ToTensor(),
])

# Use ImageFolder to load the dataset
car_dataset = ImageFolder(root=root_directory, transform=transform)

# Load the saved model
checkpoint = torch.load("car_classification_model.pth")

# Create an instance of your model class
loaded_model = CarsCNN_final()

# Load the state dictionary into your model
loaded_model.load_state_dict(checkpoint)
loaded_model.eval()
@app.route('/upload', methods=['POST'])
def upload_image():
    # Get the uploaded image file
    image_file = request.files['image']

    # Define the transformation to be applied to the images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Adjust as needed
        transforms.ToTensor(),
    ])

    # Load the image using PIL
    image = Image.open(image_file.stream)

    # Apply the transformation
    image_tensor = transform(image)

    # Add a batch dimension (required for PyTorch models)
    image_tensor = image_tensor.unsqueeze(0)

    # Classify the image
    with torch.no_grad():
        output = loaded_model(image_tensor)
        _, predicted = torch.max(output.data, 1)
        predicted_class = car_classes[predicted.item()]

    # Render the result template
    return render_template('result.html', predicted_class=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)