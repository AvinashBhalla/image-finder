import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import faiss
from PIL import Image
from torchvision.models import ResNet50_Weights

# Fix OpenMP conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.set_num_threads(1)

# Load ResNet50 model (pretrained on ImageNet)
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model.eval()  # Set to evaluation mode
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove last layer to get feature vectors

# Define Image Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize for ResNet50
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

def extract_features(image_path):
    """Extract feature vector from an image."""
    try:
        img = Image.open(image_path).convert("RGB")
        img = transform(img).unsqueeze(0)  # type: ignore # Add batch dimension
        with torch.no_grad():
            features = model(img).squeeze().numpy()  # Extract features
        return features.flatten()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def index_images(folder_path):
    """Index all images in a folder and store their features."""
    image_paths = []
    feature_list = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                features = extract_features(img_path)
                if features is not None:
                    image_paths.append(img_path)
                    feature_list.append(features)

    if len(feature_list) == 0:
        print("No valid images found in the folder.")
        return None, None

    # Convert features to a NumPy array
    feature_array = np.array(feature_list).astype("float32")

    # Use FAISS for fast nearest neighbor search
    index = faiss.IndexFlatL2(feature_array.shape[1])
    index.add(x=feature_array) # type: ignore

    return index, image_paths

def find_best_match(input_image_path, index, image_paths):
    """Find the most similar image using FAISS."""
    input_features = extract_features(input_image_path)
    if input_features is None:
        print("Could not extract features from the input image.")
        return None

    input_features = np.array([input_features]).astype("float32")
    _, best_match_index = index.search(input_features, 1)  # Find the closest match

    return image_paths[best_match_index[0][0]]

# Example Usage
search_folder = r"D:/Tk_final/ALL_PHOTOS"
input_image = r"E:\WhatsApp Image 2025-02-15 at 15.59.35_ea6c0821.jpg"

# Index all images first (only needed once)
index, image_paths = index_images(search_folder)

if index is not None:
    matched_image = find_best_match(input_image, index, image_paths)
    if matched_image:
        print(f"Most similar image found: {matched_image}")
    else:
        print("No similar image found.")
