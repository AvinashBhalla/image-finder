import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, Label, Button, StringVar
from PIL import Image, ImageTk
import threading
import random

# Load images from a folder
def load_images_from_folder(directory):
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))
    return image_paths

# Extract ORB features
def extract_features(image_path):
    orb = cv2.ORB_create() # type: ignore
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None, None
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

# Find best matching image
def find_best_match(input_image_path, folder_path):
    input_keypoints, input_descriptors = extract_features(input_image_path)
    if input_descriptors is None:
        return None, "Could not extract features."

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    best_match = None
    max_matches = 0

    for image_path in load_images_from_folder(folder_path):
        keypoints, descriptors = extract_features(image_path)
        if descriptors is None:
            continue

        matches = bf.match(input_descriptors, descriptors)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) > max_matches:
            max_matches = len(matches)
            best_match = image_path

    return best_match, "Match Found!" if best_match else "No similar image found."

# Generate positive thoughts based on images
def generate_thoughts():
    thoughts = [
        "You have a radiant personality!",
        "Your kindness reflects in your pictures.",
        "A great smile can brighten anyone’s day!",
        "Your photos capture beautiful moments of life.",
        "You have an adventurous soul, keep exploring!",
        "Every picture tells a unique story, and yours are amazing!"
    ]
    return random.choice(thoughts)

# Select input image
def select_input_image():
    global input_image_path
    input_image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if input_image_path:
        img = Image.open(input_image_path)
        img = img.resize((250, 250))
        img = ImageTk.PhotoImage(img)
        input_img_label.config(image=img)
        input_img_label.image = img # type: ignore
        input_img_label_text.config(text="Selected Image")

# Select folder
def select_folder():
    global search_folder_path
    search_folder_path = filedialog.askdirectory()
    folder_label.config(text=f"Folder: {search_folder_path}")

# Search image (with loader)
def search_image():
    if not input_image_path or not search_folder_path:
        result_label.config(text="Please select an image and folder!")
        return
    
    result_label.config(text="🔍 Searching... Please wait!")
    result_img_label.config(image='')
    thoughts_label.config(text="")
    
    def process_search():
        matched_image, message = find_best_match(input_image_path, search_folder_path)
        result_label.config(text=message)

        if matched_image:
            img = Image.open(matched_image)
            img = img.resize((250, 250))
            img = ImageTk.PhotoImage(img)
            result_img_label.config(image=img)
            result_img_label.image = img # type: ignore
            
            # Display generated thoughts
            thoughts_label.config(text=f"💡 {generate_thoughts()}")

    threading.Thread(target=process_search, daemon=True).start()

# Create GUI window
root = tk.Tk()
root.title("Image Similarity Search")
root.geometry("600x600")
root.resizable(False, False)

# Select Input Image
input_btn = Button(root, text="Select Image", command=select_input_image)
input_btn.pack(pady=5)
input_img_label_text = Label(root, text="No Image Selected")
input_img_label_text.pack()
input_img_label = Label(root)
input_img_label.pack()

# Select Folder
folder_btn = Button(root, text="Select Folder", command=select_folder)
folder_btn.pack(pady=5)
folder_label = Label(root, text="No Folder Selected")
folder_label.pack()

# Search Button
search_btn = Button(root, text="Find Similar Image", command=search_image)
search_btn.pack(pady=10)

# Display Result
result_label = Label(root, text="")
result_label.pack()
result_img_label = Label(root)
result_img_label.pack()

# Thought Generator
thoughts_label = Label(root, text="", font=("Arial", 12, "italic"), wraplength=500)
thoughts_label.pack(pady=10)

# Run GUI
input_image_path = None
search_folder_path = None
root.mainloop()
