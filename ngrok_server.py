from fastapi import FastAPI, UploadFile, File
import uvicorn
import ngrok
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from pinecone import Pinecone
import base64
from typing import List
from pydantic import BaseModel
import uuid
import io

app = FastAPI()

# Enable CORS - this is critical for frontend to backend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "https://kzmg7vb7utumdr9yr1v4.lite.vusercontent.net"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# ---- Init Pinecone ---- #
pc = Pinecone(api_key="pcsk_4Larmy_EmaoUbmwSGpwH8ree9h8zKP8kAZyhvAQGbViTtjergsFXQPdoYjV4KJGuGiYPhU")
index = pc.Index("doxxer")

# ---- Init InsightFace ---- #
face_analyzer = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_analyzer.prepare(ctx_id=0)

class Profile(BaseModel):
    id: str
    name: str
    thumbnailUrl: str
    description: str
    faceBox: dict  # { top: string, left: string, width: string, height: string }

@app.get("/")
def read_root():
    return {"message": "Hello World"}

@app.post("/analyze", response_model=List[Profile])
async def analyze_image(file: UploadFile = File(...)):
    # Read image from uploaded file
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return {"error": "Invalid image data"}
    
    # Get image dimensions for percentage calculations
    img_height, img_width = img.shape[:2]
    
    # Detect faces
    faces = face_analyzer.get(img)
    
    profiles = []
    
    # Process each detected face
    for face in faces:
        # Get bounding box
        box = face.bbox.astype(int)
        x1, y1, x2, y2 = box
        
        # Calculate box as percentages
        face_box = {
            "top": f"{(y1 / img_height) * 100}%",
            "left": f"{(x1 / img_width) * 100}%",
            "width": f"{((x2 - x1) / img_width) * 100}%",
            "height": f"{((y2 - y1) / img_height) * 100}%"
        }
        
        # Extract embedding and query database
        embedding = face.normed_embedding.tolist()
        results = index.query(
            vector=embedding,
            top_k=1,
            include_metadata=True,
            include_values=False
        )
        
        # Get match information
        match = results.get("matches", [])[0] if results.get("matches", []) else None
        
        if match:
            metadata = match["metadata"]
            name = metadata.get("name", "Unknown")
            description = f"Confidence: {match['score']:.2f}"
            
            # Create a thumbnail from the face region
            face_img = img[y1:y2, x1:x2]
            _, buffer = cv2.imencode('.jpg', face_img)
            thumbnail_b64 = base64.b64encode(buffer).decode('utf-8')
            thumbnail_url = f"data:image/jpeg;base64,{thumbnail_b64}"
            
            profile = Profile(
                id=str(uuid.uuid4()),
                name=name,
                thumbnailUrl=thumbnail_url,
                description=description,
                faceBox=face_box
            )
            
            profiles.append(profile)
    
    return profiles

if __name__ == "__main__":
    # Start ngrok tunnel
    listener = ngrok.forward(addr="8000", authtoken_from_env=True)
    print(f"Ingress established at: {listener.url()}")
    
    # Start uvicorn server
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")