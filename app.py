import os
import torch
import faiss
import numpy as np
import json
import gradio as gr
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

DATABASE_FOLDER = "images"
INDEX_FILE = "vendor_images.index"
MAPPING_FILE = "image_mapping.json"
DIMENSION = 512

# ── 1. Load model once at startup ──────────────────────────────────────────
print("Loading CLIP model... (first run downloads ~600MB)")
model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)
print("-> Model loaded.")


def extract_features(output):
    """Normalize whichever object type get_*_features() returns into a plain tensor."""
    if torch.is_tensor(output):
        return output
    if hasattr(output, 'image_embeds') and output.image_embeds is not None:
        return output.image_embeds
    if hasattr(output, 'text_embeds') and output.text_embeds is not None:
        return output.text_embeds
    if hasattr(output, 'pooler_output') and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, 'last_hidden_state') and output.last_hidden_state is not None:
        return output.last_hidden_state
    raise TypeError(f"Unexpected output type: {type(output)}")


def embed_images(images):
    """Turn a list of PIL images into normalized CLIP embeddings (numpy, float32)."""
    inputs = processor(images=images, return_tensors="pt")
    with torch.no_grad():
        out = model.get_image_features(**inputs)
        features = extract_features(out).float()
        features /= features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().astype(np.float32)


def embed_text(text):
    """Turn a text query into a normalized CLIP embedding (numpy, float32)."""
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model.get_text_features(**inputs)
        features = extract_features(out).float()
        features /= features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().astype(np.float32)


# ── 2. Load or build the FAISS index ONCE at startup ───────────────────────
def load_or_build_index():
    if os.path.exists(INDEX_FILE) and os.path.exists(MAPPING_FILE):
        print(f"-> Loading existing index '{INDEX_FILE}'...")
        index = faiss.read_index(INDEX_FILE)
        with open(MAPPING_FILE, "r") as f:
            image_names = json.load(f)
        print(f"-> Loaded {index.ntotal} indexed images.")
    else:
        print("-> No existing index found. Starting fresh.")
        index = faiss.IndexFlatIP(DIMENSION)
        image_names = []

    sync_new_images(index, image_names)
    return index, image_names


def sync_new_images(index, image_names):
    """Embed and add only images that aren't already in the index. Mutates index/image_names in place."""
    if not os.path.exists(DATABASE_FOLDER):
        return

    on_disk = sorted(
        f for f in os.listdir(DATABASE_FOLDER)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    )
    existing = set(image_names)
    new_files = [f for f in on_disk if f not in existing]

    if not new_files:
        return

    print(f"-> Found {len(new_files)} new image(s), embedding: {new_files}")
    new_images = [Image.open(os.path.join(DATABASE_FOLDER, f)).convert("RGB") for f in new_files]
    new_vectors = embed_images(new_images)

    index.add(new_vectors)
    image_names.extend(new_files)

    faiss.write_index(index, INDEX_FILE)
    with open(MAPPING_FILE, "w") as f:
        json.dump(image_names, f)
    print(f"-> Index updated. Now has {index.ntotal} images.")


# Build/load the index once when the app starts (not on every search)
index, image_names = load_or_build_index()


# ── 3. Search functions — query the index directly, no recomputation ───────
def search_index(query_vector, top_k=12):
    """query_vector: numpy array shape (1, 512). Returns list of (filename, score)."""
    if index.ntotal == 0:
        return []
    k = min(top_k, index.ntotal)
    scores, ids = index.search(query_vector, k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        results.append((image_names[idx], float(score)))
    return results


def to_gallery(results):
    gallery_items = []
    for name, score in results:
        path = os.path.join(DATABASE_FOLDER, name)
        gallery_items.append((path, f"{name}  ({score:.4f})"))
    return gallery_items


def search_by_text(query):
    if not query or not query.strip():
        return [], "Type a description to search."

    # Pick up any images added to the folder since last search — cheap check,
    # only embeds files that are actually new.
    sync_new_images(index, image_names)

    if index.ntotal == 0:
        return [], f"No images found in the '{DATABASE_FOLDER}' folder."

    text_vector = embed_text(query)
    results = search_index(text_vector)
    return to_gallery(results), f"Searched {index.ntotal} images. Showing best matches for: \"{query}\""


def search_by_image(uploaded_image):
    if uploaded_image is None:
        return [], "Upload an image to search."

    sync_new_images(index, image_names)

    if index.ntotal == 0:
        return [], f"No images found in the '{DATABASE_FOLDER}' folder."

    uploaded_image = uploaded_image.convert("RGB")
    query_vector = embed_images([uploaded_image])
    results = search_index(query_vector)
    return to_gallery(results), f"Searched {index.ntotal} images. Showing best visual matches."


# ── 4. Build the UI ─────────────────────────────────────────────────────────
with gr.Blocks(title="CLIP Image Search") as demo:
    gr.Markdown("# 🔍 AI Image Search")
    gr.Markdown(
        f"Searches images inside the **`{DATABASE_FOLDER}/`** folder using a persistent FAISS index. "
        "Embeddings are computed once and reused — only newly added images get embedded."
    )

    with gr.Tab("Search by Text"):
        text_input = gr.Textbox(
            label="Describe what you're looking for",
            placeholder="e.g. Birthday theme with grey, white colours and including animals",
        )
        text_button = gr.Button("Search", variant="primary")
        text_status = gr.Markdown()
        text_gallery = gr.Gallery(label="Results", columns=4, height="auto")

        text_button.click(fn=search_by_text, inputs=text_input, outputs=[text_gallery, text_status])
        text_input.submit(fn=search_by_text, inputs=text_input, outputs=[text_gallery, text_status])

    with gr.Tab("Search by Image"):
        image_input = gr.Image(label="Upload a search image", type="pil")
        image_button = gr.Button("Search", variant="primary")
        image_status = gr.Markdown()
        image_gallery = gr.Gallery(label="Results", columns=4, height="auto")

        image_button.click(fn=search_by_image, inputs=image_input, outputs=[image_gallery, image_status])


if __name__ == "__main__":
    demo.launch()