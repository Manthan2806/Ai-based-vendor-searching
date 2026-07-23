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
    raise TypeError(f"Unexpected output type: {type(output)}")


def embed_images(images):
    inputs = processor(images=images, return_tensors="pt")
    with torch.no_grad():
        out = model.get_image_features(**inputs)
        features = extract_features(out).float()
        features /= features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().astype(np.float32)


def embed_text(text):
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model.get_text_features(**inputs)
        features = extract_features(out).float()
        features /= features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().astype(np.float32)


# ── 2. Load or build the FAISS index ONCE at startup ───────────────────────
def load_or_build_index():
    if os.path.exists(INDEX_FILE) and os.path.exists(MAPPING_FILE):
        index = faiss.read_index(INDEX_FILE)
        with open(MAPPING_FILE, "r") as f:
            image_names = json.load(f)
    else:
        index = faiss.IndexFlatIP(DIMENSION)
        image_names = []

    sync_new_images(index, image_names)
    return index, image_names


def sync_new_images(index, image_names):
    if not os.path.exists(DATABASE_FOLDER):
        os.makedirs(DATABASE_FOLDER)
        return

    on_disk = sorted(f for f in os.listdir(DATABASE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg')))
    existing = set(image_names)
    new_files = [f for f in on_disk if f not in existing]

    if not new_files:
        return

    print(f"-> Indexing {len(new_files)} new image(s)...")
    new_images = [Image.open(os.path.join(DATABASE_FOLDER, f)).convert("RGB") for f in new_files]
    new_vectors = embed_images(new_images)

    index.add(new_vectors)
    image_names.extend(new_files)

    faiss.write_index(index, INDEX_FILE)
    with open(MAPPING_FILE, "w") as f:
        json.dump(image_names, f)


index, image_names = load_or_build_index()


def search_index(query_vector, top_k=4):
    """Searches FAISS and returns a list of (filename, score)."""
    if index.ntotal == 0:
        return []
    k = min(top_k, index.ntotal)
    scores, ids = index.search(query_vector, k)

    # Only return results with a score above 0.20 to filter out junk
    return [(image_names[idx], float(score)) for score, idx in zip(scores[0], ids[0]) if idx != -1 and score > 0.20]


# ── 3. Chatbot Interaction Logic (Gradio 5 Messages API) ────────────────────
def chat_interaction(message, history):
    """
    Handles multimodal chat inputs for Gradio 5+.
    `message` is a dictionary: {'text': str, 'files': list}
    """
    text_query = message.get("text", "").strip()
    files = message.get("files", [])

    # Safely extract image path (handles both string paths and file dicts)
    image_path = None
    if files:
        first_file = files[0]
        image_path = first_file.get("path") if isinstance(first_file, dict) else str(first_file)

    # If user sent nothing
    if not text_query and not image_path:
        history.append({"role": "assistant", "content": "Please type a description or upload an image to search."})
        return history, gr.MultimodalTextbox(value=None, interactive=True)

    # 1. Format user inputs in history
    # NOTE: image content must be a tuple (filepath,) -- that's what tells
    # Gradio's Chatbot to render it as an image bubble instead of text.
    if image_path:
        history.append({"role": "user", "content": {"path": image_path}})
    if text_query:
        history.append({"role": "user", "content": text_query})

    # 2. Check for new images in the background before searching
    sync_new_images(index, image_names)

    # 3. Calculate vectors
    text_vec = embed_text(text_query) if text_query else None
    img_vec = embed_images([Image.open(image_path).convert("RGB")]) if image_path else None

    # 4. Multimodal fusion math
    alpha = 0.6  # 60% text weight, 40% image weight

    if text_vec is not None and img_vec is not None:
        final_vec = (alpha * text_vec) + ((1 - alpha) * img_vec)
    elif text_vec is not None:
        final_vec = text_vec
    else:
        final_vec = img_vec

    # Re-normalize vector
    final_vec /= np.linalg.norm(final_vec, axis=-1, keepdims=True)
    final_vec = final_vec.astype(np.float32)

    # 5. Search FAISS index
    results = search_index(final_vec, top_k=4)

    # 6. Format bot responses in history
    if not results:
        history.append({"role": "assistant", "content": "I couldn't find any close matches for that in the database."})
    else:
        history.append({"role": "assistant", "content": "Here are the closest matches from your vendor database:"})
        for name, score in results:
            full_img_path = os.path.join(DATABASE_FOLDER, name)

            # Image bubble -- dict with "path" key is the format Gradio 6 expects
            history.append({"role": "assistant", "content": {"path": full_img_path}})
            # Score label as plain text
            history.append({"role": "assistant", "content": f"Match Score: {score:.2f} ({name})"})

    # Return updated history and clear the textbox
    return history, gr.MultimodalTextbox(value=None, interactive=True)


# ── 4. Build the UI ─────────────────────────────────────────────────────────
with gr.Blocks(title="AI Vendor Search") as demo:
    gr.Markdown("# 🔍 Event Vendor AI Search")
    gr.Markdown("Type a description, drag in a reference photo, or **do both at the same time** to filter vendors.")

    # Gradio 6.x uses the dict-based {"role":..., "content":...} message format
    # by default -- no `type=` argument needed anymore.
    chatbot = gr.Chatbot(height=600, label="Search Results")

    chat_input = gr.MultimodalTextbox(
        interactive=True,
        file_types=["image"],
        placeholder="Type a request or upload an image...",
        show_label=False
    )

    chat_input.submit(
        fn=chat_interaction,
        inputs=[chat_input, chatbot],
        outputs=[chatbot, chat_input]
    )

if __name__ == "__main__":
    demo.launch()