import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# 1. Force visual confirmation that the script started
print("\n=========================================")
print("SUCCESS: The script has officially started!")
print("=========================================\n")

print("Loading the AI model... If this is your first time, it will download 600MB.")
print("Please wait...")

model_id = "openai/clip-vit-base-patch32"

model = CLIPModel.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)

print("\n-> Model loaded successfully!")

# 2. Setup images
search_image_path = "searchimage.jpg"
database_folder = "images"

if not os.path.exists(search_image_path):
    print(f"ERROR: Cannot find '{search_image_path}' in this folder!")
    exit()

if not os.path.exists(database_folder):
    print(f"ERROR: Cannot find the '{database_folder}' folder!")
    exit()

search_image = Image.open(search_image_path)
database_images = []
image_names = []

for filename in os.listdir(database_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        path = os.path.join(database_folder, filename)
        database_images.append(Image.open(path))
        image_names.append(filename)

if len(database_images) == 0:
    print(f"ERROR: The '{database_folder}' folder is empty! Put some JPG images inside it.")
    exit()

print(f"-> Successfully loaded {len(database_images)} database images.")


def extract_features(output):
    """
    Different transformers versions return different object types from
    get_image_features() / get_text_features(). This normalizes all of
    them down to a plain tensor.
    """
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


# 3. Process images and text
print("-> Comparing images using AI math...")

# A. Calculate Image Features for the database
image_inputs = processor(images=database_images, return_tensors="pt")
with torch.no_grad():
    img_out = model.get_image_features(**image_inputs)
    image_features = extract_features(img_out).float()
    image_features /= image_features.norm(dim=-1, keepdim=True)

# B. Calculate Text Features for the query
text_query = 'Birthday theme with grey, white colours and strictly including animals'
text_inputs = processor(text=[text_query], return_tensors="pt", padding=True)
with torch.no_grad():
    txt_out = model.get_text_features(**text_inputs)
    text_features = extract_features(txt_out).float()
    text_features /= text_features.norm(dim=-1, keepdim=True)

# C. Compare the text vector to all image vectors
similarity_scores = torch.nn.functional.cosine_similarity(text_features, image_features)

# D. Pair and sort
results = list(zip(image_names, similarity_scores.tolist()))
results.sort(key=lambda x: x[1], reverse=True)

print("\n-> Results (most similar first):")
for name, score in results:
    print(f"   {name}: {score:.4f}")