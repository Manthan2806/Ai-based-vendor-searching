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

# Added local_files_only=False explicitly to trigger downloading logs
model = CLIPModel.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)

print("\n-> Model loaded successfully!")

# 2. Setup images
search_image_path = "searchimage.jpg"
database_folder = "images"

# Check if search image exists
if not os.path.exists(search_image_path):
    print(f"ERROR: Cannot find '{search_image_path}' in this folder!")
    exit()

# Check if images folder exists
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

# 3. Process images

print("-> Comparing images using AI math...")
inputs = processor(images=[search_image] + database_images, return_tensors="pt")

with torch.no_grad():
    output = model.get_image_features(**inputs)

    if torch.is_tensor(output):
        image_features = output
    elif hasattr(output, 'image_embeds') and output.image_embeds is not None:
        image_features = output.image_embeds
    elif hasattr(output, 'pooler_output') and output.pooler_output is not None:
        image_features = output.pooler_output
    else:
        raise TypeError(f"Unexpected output type from get_image_features: {type(output)}")

# Now we perform the math on the extracted tensor
# .float() ensures the numbers are in the right format for torch.cosine_similarity
image_features = image_features.float()

# Normalization makes the comparison math much more accurate
image_features /= image_features.norm(dim=-1, keepdim=True)

search_vector = image_features[0].unsqueeze(0)
database_vectors = image_features[1:]

similarity_scores = torch.nn.functional.cosine_similarity(search_vector, database_vectors)

print(similarity_scores)

# Pair each score with its filename and sort by similarity (highest first)
results = list(zip(image_names, similarity_scores.tolist()))
results.sort(key=lambda x: x[1], reverse=True)

print("\n-> Results (most similar first):")
for name, score in results:
    print(f"   {name}: {score:.4f}")