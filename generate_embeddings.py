import os
import glob
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model

def build_pretrained_resnet_extractor():
    """
    Returns a Keras model that outputs feature vectors (embeddings)
    from a pretrained ResNet50 (ImageNet).
    We'll remove the top classification head and apply global average pooling.
    """
    base_model = ResNet50(weights='imagenet', include_top=False)
    x = GlobalAveragePooling2D()(base_model.output)  # shape: (None, 2048)
    model = Model(inputs=base_model.input, outputs=x)
    return model

def preprocess_img(img_path, target_size=(224, 224)):
    """
    Load & preprocess an image for ResNet50 (ImageNet).
    """
    img = Image.open(img_path).convert("RGB").resize(target_size)
    arr = np.array(img)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

def generate_embeddings(data_folder, embeddings_file, paths_file):
    """
    1) Find images in data_folder (recursively).
    2) Use pretrained ResNet50 to get embeddings.
    3) Save them in embeddings_file & paths_file for later use.
    """
    # Gather all images under data_folder
    exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    all_paths = []
    for ext in exts:
        # search data_folder and its subfolders
        all_paths.extend(glob.glob(os.path.join(data_folder, ext)))
        all_paths.extend(glob.glob(os.path.join(data_folder, "**", ext), recursive=True))

    all_paths = list(set(all_paths))  # remove duplicates if any
    if not all_paths:
        print(f"No images found in {data_folder}.")
        return

    # Build the feature extractor
    extractor = build_pretrained_resnet_extractor()

    embeddings_list = []
    valid_paths = []
    for path in all_paths:
        try:
            x = preprocess_img(path)
            emb = extractor.predict(x)[0]  # shape (2048,)
            embeddings_list.append(emb)

            # Store as relative path, relative to the project directory
            # so that `app.py` can rejoin correctly later.
            project_dir = os.path.dirname(os.path.abspath(__file__))
            rel_path = os.path.relpath(path, start=project_dir)
            valid_paths.append(rel_path)
        except Exception as e:
            print(f"Error with {path}: {e}")

    if not embeddings_list:
        print("No valid embeddings generated.")
        return

    embeddings = np.array(embeddings_list, dtype='float32')
    valid_paths = np.array(valid_paths, dtype=object)

    # Save to .npy
    np.save(embeddings_file, embeddings)
    np.save(paths_file, valid_paths)
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Saved embeddings to {embeddings_file}")
    print(f"Saved image paths to {paths_file}")

if __name__ == "__main__":
    # Paths
    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(project_dir, "data")  # subfolder with images
    embeddings_file = os.path.join(project_dir, "image_embeddings.npy")
    paths_file = os.path.join(project_dir, "image_paths.npy")

    generate_embeddings(data_folder, embeddings_file, paths_file)
