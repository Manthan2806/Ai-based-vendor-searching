# from fastapi import FastAPI
# from pydantic import BaseModel
# from fastapi import FastAPI, File, UploadFile

# app = FastAPI() # This creates your "waiter"

# # When a GET request hits the home URL ("/"), run this function
# @app.get("/")
# def home():
#     return {"message": "My first backend is running!"}



# class SearchRequest(BaseModel):
#     query: str
#     budget: int

# # 2. Create the POST route
# @app.post("/search")
# def search_vendors(data: SearchRequest):
    
#     # 3. Use the data! 
#     # FastAPI automatically checks that budget is a number before this code even runs.
#     search_term = data.query
#     max_price = data.budget
    
#     # Send an answer back to React
#     return {
#         "status": "success",
#         "message": f"Searching for {search_term} under {max_price} rupees."
#     }



# @app.post("/upload-image")
# def receive_image(image: UploadFile = File(...)):
#     # Extract basic information about the uploaded file
#     file_name = image.filename
#     file_type = image.content_type
    
#     # In the future, this is where you will pass the image to PyTorch
    
#     return {
#         "status": "success",
#         "message": "Image received!",
#         "filename": file_name,
#         "type": file_type
#     }


import faiss
import torch
from transformers import CLIPProcessor, CLIPModel
from contextlib import asynccontextmanager
from typing import Annotated
from fastapi import FastAPI, File, Form, Request, UploadFile


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP ---
    print("Initializing Occasia backend resources...")
    
    # 1. Load CLIP Model
    model_id = "openai/clip-vit-base-patch32"
    app.state.clip_model = CLIPModel.from_pretrained(model_id)
    app.state.clip_processor = CLIPProcessor.from_pretrained(model_id)
    
    # 2. Load FAISS Index
    app.state.faiss_index = faiss.read_index("vendor_images.index")
    
    yield # Server is running and handling incoming requests
    
    # --- SHUTDOWN ---
    print("Shutting down and cleaning up memory...")
    del app.state.clip_model
    del app.state.clip_processor
    del app.state.faiss_index
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan)

from fastapi.middleware.cors import CORSMiddleware

# Add right after initializing app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite + React default port
        "http://localhost:3000",  # Create React App default port
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Allows GET, POST, OPTIONS, etc.
    allow_headers=["*"],  # Allows headers like Content-Type
)
import io
from typing import Optional
import torch
from PIL import Image
from fastapi import FastAPI, File, Form, Request, UploadFile

@app.post("/api/search")
async def search_vendors(
    request: Request,
    query: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None)
):
    model = request.app.state.clip_model
    processor = request.app.state.clip_processor
    index = request.app.state.faiss_index

    # Guard clause: Ensure at least one search input was sent
    if not query and not image:
        return {"status": "error", "message": "Please provide a query string, an image file, or both."}

    embeddings = []

    # 1. Process Text Input (if present)
    if query:
        text_inputs = processor(text=[query], return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model.get_text_features(**text_inputs)
            text_tensor = outputs.text_embeds if hasattr(outputs, "text_embeds") else outputs
            # L2 Normalize text vector
            text_tensor = text_tensor / text_tensor.norm(p=2, dim=-1, keepdim=True)
            embeddings.append(text_tensor)

    # 2. Process Image Input (if present)
    if image:
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_inputs = processor(images=pil_image, return_tensors="pt")
        with torch.no_grad():
            outputs = model.get_image_features(**image_inputs)
            image_tensor = outputs.image_embeds if hasattr(outputs, "image_embeds") else outputs
            # L2 Normalize image vector
            image_tensor = image_tensor / image_tensor.norm(p=2, dim=-1, keepdim=True)
            embeddings.append(image_tensor)

    # 3. Vector Fusion
    if len(embeddings) == 2:
        # Combine normalized text + image vectors and re-normalize
        combined_tensor = (embeddings[0] + embeddings[1]) / 2.0
        combined_tensor = combined_tensor / combined_tensor.norm(p=2, dim=-1, keepdim=True)
    else:
        combined_tensor = embeddings[0]

    # 4. Convert vector to float32 NumPy array for FAISS
    search_vector = combined_tensor.cpu().detach().numpy().astype("float32")

    # 5. Execute FAISS similarity search
    distances, indices = index.search(search_vector, k=5)

    return {
        "status": "success",
        "matches": indices.tolist()[0],
        "distances": distances.tolist()[0]
    }