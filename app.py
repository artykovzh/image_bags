import streamlit as st
import os
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model

@st.cache_data
def load_embeddings_and_paths(emb_file, paths_file):
    """
    Load the precomputed embeddings and file paths from .npy.
    """
    embeddings = np.load(emb_file).astype('float32')
    image_paths = np.load(paths_file, allow_pickle=True)
    return embeddings, image_paths

@st.cache_resource
def build_resnet_feature_extractor():
    """
    Build a pretrained ResNet50 (ImageNet) feature extractor:
    - remove top layers
    - global average pooling
    """
    base_model = ResNet50(weights='imagenet', include_top=False)
    x = GlobalAveragePooling2D()(base_model.output)
    model = Model(inputs=base_model.input, outputs=x)
    return model

def preprocess_image_for_resnet(img, target_size=(224,224)):
    """
    Preprocess a user-uploaded image for ResNet50.
    """
    img = img.convert("RGB").resize(target_size)
    arr = np.array(img)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

def cosine_sim_recommendations(query_embedding, embeddings, top_k=5):
    """
    Compute cosine similarity between query_embedding and all embeddings.
    Return top_k indices and scores in descending order.
    """
    # query_embedding => (1, 2048), embeddings => (N, 2048)
    sims = cosine_similarity(query_embedding, embeddings)[0]  # shape [N]
    # Sort by similarity descending
    top_indices = np.argsort(sims)[::-1][:top_k]
    return [(idx, sims[idx]) for idx in top_indices]

def main():
    st.title("🖼️ Image Based Search")

    # 1. Load the embeddings & image paths
    project_dir = os.path.dirname(os.path.abspath(__file__))
    emb_file = os.path.join(project_dir, "image_embeddings.npy")
    paths_file = os.path.join(project_dir, "image_paths.npy")

    if not (os.path.exists(emb_file) and os.path.exists(paths_file)):
        st.error("Embeddings or paths files not found. Please run generate_embeddings.py first.")
        return

    embeddings, image_paths = load_embeddings_and_paths(emb_file, paths_file)

    # 2. Build the pretrained ResNet50 feature extractor
    extractor = build_resnet_feature_extractor()

    # 3. User uploads an image
    uploaded_file = st.file_uploader("Upload an image (jpg, jpeg, png)...", type=["jpg","jpeg","png"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", width=300)

        # 4. Embed the uploaded image
        img = Image.open(uploaded_file)
        query_tensor = preprocess_image_for_resnet(img)
        query_emb = extractor.predict(query_tensor).astype('float32')  # shape (1, 2048)

        # 5. Find top-k similar images by cosine similarity
        k = 5
        results = cosine_sim_recommendations(query_emb, embeddings, top_k=k)

        # 6. Display the results, two images per row
        st.subheader("Top Recommendations")
        for i in range(0, len(results), 2):
            cols = st.columns(2)
            for col_index in range(2):
                if i + col_index < len(results):
                    idx, score = results[i + col_index]
                    with cols[col_index]:
                        st.markdown(f"**Rank {i + col_index + 1}, Similarity = {score:.3f}**")
                        matched_path = image_paths[idx]
                        st.image(matched_path, width=300)

if __name__ == "__main__":
    main()
