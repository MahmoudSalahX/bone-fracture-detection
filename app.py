import gradio as gr
import torch
from PIL import Image
import torchvision.transforms as transforms
from model import create_model # We import the model structure from model.py

# --- 1. Configuration ---
# Set the device to CPU (recommended for inference on free hosting)
device = torch.device('cpu')
# The name of your trained model file
MODEL_PATH = 'bone_fracture_model_final.pth'
# Define the class names
class_names = ['fracture', 'normal']

# --- 2. Load the Trained Model ---
print("Loading model...")
# Create a new instance of the model architecture
model = create_model(device)
# Load the saved weights into the model
# Use map_location=device to ensure it loads correctly on a CPU
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval() # Set the model to evaluation mode
print("Model loaded successfully!")

# --- 3. Define Image Transformations ---
inference_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 4. Create the Prediction Function ---
def predict(image):
    """
    Takes a PIL image, processes it, and returns the model's predictions.
    """
    # The input 'image' from Gradio is a PIL Image. Convert to RGB.
    image = image.convert('RGB')
    
    # Apply transformations and add a batch dimension
    image_tensor = inference_transforms(image).unsqueeze(0).to(device)
    
    # Make a prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
    # Create a dictionary of labels and their confidence scores
    confidences = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
    return confidences

# --- 5. Launch the Gradio Interface ---
gr.Interface(fn=predict, 
             inputs=gr.Image(type="pil", label="Upload X-Ray Image"),
             outputs=gr.Label(num_top_classes=2, label="Prediction Results"),
             title="Bone Fracture Detector",
             description="An AI model to detect bone fractures from X-ray images. This model was trained on a public dataset and achieved 100% accuracy on its test set.",
             examples=[
                 # You can add example images here if you upload them to your Space
             ]).launch()

