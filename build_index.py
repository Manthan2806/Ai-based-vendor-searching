import os
import torch
import faiss
import numpy as np
import json
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

DATABASE_FOLDER = "images"
INDEX_FILE = "vendor_images.index"
MAPPING_FILE = "image_mapping.json"
DIMENSION = 512


def extract_features(output):
    """
    Different transformers versions return different object types from
    get_image_features(). This normalizes all of them down to a plain tensor.
    """
    if torch.is_tensor(output):
        return output
    if hasattr(output, 'image_embeds') and output.image_embeds is not None:
        return output.image_embeds
    if hasattr(output, 'pooler_output') and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, 'last_hidden_state') and output.last_hidden_state is not None:
        return output.last_hidden_state
    raise TypeError(f"Unexpected output type: {type(output)}")


def load_existing_index_and_mapping():
    """Load a previously saved index + filename list, or start fresh if none exist."""
    if os.path.exists(INDEX_FILE) and os.path.exists(MAPPING_FILE):
        print(f"-> Found existing database. Loading '{INDEX_FILE}'...")
        index = faiss.read_index(INDEX_FILE)
        with open(MAPPING_FILE, "r") as f:
            image_names = json.load(f)
        print(f"-> Loaded {index.ntotal} previously indexed images.")
        return index, image_names
    else:
        print("-> No existing database found. Starting fresh.")
        index = faiss.IndexFlatIP(DIMENSION)
        image_names = []
        return index, image_names


def find_new_images(existing_names):
    """Compare filenames on disk vs. what's already in the index."""
    all_files_on_disk = [
        f for f in sorted(os.listdir(DATABASE_FOLDER))
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    existing_set = set(existing_names)
    new_files = [f for f in all_files_on_disk if f not in existing_set]
    return new_files


def embed_images(model, processor, images):
    """Turn a list of PIL images into normalized CLIP embeddings (numpy, float32)."""
    inputs = processor(images=images, return_tensors="pt")
    with torch.no_grad():
        out = model.get_image_features(**inputs)
        features = extract_features(out).float()
        features /= features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().astype(np.float32)


def main():
    if not os.path.exists(DATABASE_FOLDER):
        print(f"ERROR: '{DATABASE_FOLDER}' folder not found!")
        return

    # 1. Load whatever already exists (or start empty)
    index, image_names = load_existing_index_and_mapping()

    # 2. Figure out which files on disk are NOT yet in the index
    new_files = find_new_images(image_names)

    if not new_files:
        print("-> No new images found. Database is already up to date.")
        return

    print(f"-> Found {len(new_files)} new image(s) to add: {new_files}")

    # 3. Load the AI model (only needed if there's actually new work to do)
    print("-> Loading AI model...")
    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id)

    # 4. Open + embed ONLY the new images
    new_images = [Image.open(os.path.join(DATABASE_FOLDER, f)) for f in new_files]
    print("-> Calculating vectors for new images only...")
    new_vectors = embed_images(model, processor, new_images)

    # 5. Append new vectors to the existing index (old vectors untouched)
    index.add(new_vectors)
    image_names.extend(new_files)

    # 6. Save the updated index + mapping back to disk
    faiss.write_index(index, INDEX_FILE)
    with open(MAPPING_FILE, "w") as f:
        json.dump(image_names, f)

    print(f"\nSUCCESS: Added {len(new_files)} new image(s). "
          f"Database now has {index.ntotal} total images.")


if __name__ == "__main__":
    main()