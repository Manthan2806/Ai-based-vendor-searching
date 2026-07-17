import os
import torch
import gradio as gr
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# ── 1. Load model once at startup ──────────────────────────────────────────
print("Loading CLIP model... (first run downloads ~600MB)")
model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)
print("-> Model loaded.")

DATABASE_FOLDER = "images"


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


def load_database_images():
    """Load every image in the database folder + their filenames."""
    images, names = [], []
    if not os.path.exists(DATABASE_FOLDER):
        return images, names
    for filename in sorted(os.listdir(DATABASE_FOLDER)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(DATABASE_FOLDER, filename)
            images.append(Image.open(path).convert("RGB"))
            names.append(filename)
    return images, names


def embed_images(images):
    """Turn a list of PIL images into normalized CLIP embeddings."""
    inputs = processor(images=images, return_tensors="pt")
    with torch.no_grad():
        out = model.get_image_features(**inputs)
        features = extract_features(out).float()
        features /= features.norm(dim=-1, keepdim=True)
    return features


def embed_text(text):
    """Turn a text query into a normalized CLIP embedding."""
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model.get_text_features(**inputs)
        features = extract_features(out).float()
        features /= features.norm(dim=-1, keepdim=True)
    return features


def rank_results(query_features, db_features, names, db_images, top_k=12):
    """Cosine similarity + sort, returns gallery-ready list of (image, caption)."""
    scores = torch.nn.functional.cosine_similarity(query_features, db_features)
    ranked = sorted(zip(names, scores.tolist(), db_images), key=lambda x: x[1], reverse=True)
    ranked = ranked[:top_k]
    return [(img, f"{name}  ({score:.4f})") for name, score, img in ranked]


# ── 2. Search functions wired to the UI ─────────────────────────────────────
def search_by_text(query):
    if not query or not query.strip():
        return [], "Type a description to search."

    db_images, db_names = load_database_images()
    if not db_images:
        return [], f"No images found in the '{DATABASE_FOLDER}' folder."

    text_features = embed_text(query)
    db_features = embed_images(db_images)
    gallery_items = rank_results(text_features, db_features, db_names, db_images)
    return gallery_items, f"Found {len(db_images)} images. Showing best matches for: \"{query}\""


def search_by_image(uploaded_image):
    if uploaded_image is None:
        return [], "Upload an image to search."

    db_images, db_names = load_database_images()
    if not db_images:
        return [], f"No images found in the '{DATABASE_FOLDER}' folder."

    uploaded_image = uploaded_image.convert("RGB")
    query_features = embed_images([uploaded_image])  # batch of 1
    db_features = embed_images(db_images)
    gallery_items = rank_results(query_features, db_features, db_names, db_images)
    return gallery_items, f"Found {len(db_images)} images. Showing best visual matches."


# ── 3. Build the UI ─────────────────────────────────────────────────────────
with gr.Blocks(title="CLIP Image Search") as demo:
    gr.Markdown("# 🔍 AI Image Search")
    gr.Markdown(
        f"Searches images inside the **`{DATABASE_FOLDER}/`** folder using CLIP embeddings. "
        "Search by describing what you want, or upload an image to find visually similar ones."
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