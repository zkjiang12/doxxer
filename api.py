from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from pinecone import Pinecone
import base64
from typing import List, Optional
from pydantic import BaseModel
import uuid
import io

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    
class UserRegistration(BaseModel):
    name: str
    description: str

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
        matches = results.get("matches", [])
        match = matches[0] if matches else None
        
        if match and match["score"] > 0.38:
            metadata = match["metadata"]
            name = metadata.get("name", "Unknown")
            person_description = metadata.get("description", "")
            confidence = f"Confidence: {match['score']:.2f}"
            description = f"{person_description}\n{confidence}" if person_description else confidence
            
            # Get image URL from Pinecone metadata instead of using face crop
            thumbnail_url = metadata.get("img_url", "")
            
            # If no image URL in metadata, fallback to face crop
            if not thumbnail_url:
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
        else:
            # No match or match below threshold
            # Create thumbnail from face crop
            # Make sure coordinates are valid
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_width, x2)
            y2 = min(img_height, y2)
            
            # Check if we have a valid face crop
            if x1 < x2 and y1 < y2:
                face_img = img[y1:y2, x1:x2]
                _, buffer = cv2.imencode('.jpg', face_img)
                thumbnail_b64 = base64.b64encode(buffer).decode('utf-8')
                thumbnail_url = f"data:image/jpeg;base64,{thumbnail_b64}"
            else:
                # If invalid crop, use a placeholder or the full image
                thumbnail_url = ""
            
            profile = Profile(
                id=str(uuid.uuid4()),
                name="RIP no match",
                thumbnailUrl=thumbnail_url,
                description="Confidence too low or no match found",
                faceBox=face_box
            )
            
            profiles.append(profile)
    
    return profiles

@app.post("/register")
async def register_user(
    file: UploadFile = File(...),
    name: str = Form(...),
    description: str = Form(...)
):
    # Read image from uploaded file
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image data")
    
    # Detect faces
    faces = face_analyzer.get(img)
    
    if not faces:
        raise HTTPException(status_code=400, detail="No face detected in the image")
    
    if len(faces) > 1:
        raise HTTPException(status_code=400, detail="Multiple faces detected. Please upload an image with only one face.")
    
    # Get the detected face
    face = faces[0]
    
    # Extract embedding
    embedding = face.normed_embedding.tolist()
    
    # Create a thumbnail from the face
    box = face.bbox.astype(int)
    x1, y1, x2, y2 = box
    
    # Make sure coordinates are valid
    img_height, img_width = img.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_width, x2)
    y2 = min(img_height, y2)
    
    # Check if we have a valid face crop
    if x1 >= x2 or y1 >= y2:
        raise HTTPException(status_code=400, detail="Invalid face detection")
    
    # Create face image
    face_img = img[y1:y2, x1:x2]
    _, buffer = cv2.imencode('.jpg', face_img)
    img_b64 = base64.b64encode(buffer).decode('utf-8')
    img_url = f"data:image/jpeg;base64,{img_b64}"
    
    # Create unique ID
    user_id = str(uuid.uuid4())
    
    # Store in Pinecone
    index.upsert(
        vectors=[
            {
                "id": user_id,
                "values": embedding,
                "metadata": {
                    "name": name,
                    "description": description,
                    "img_url": img_url
                }
            }
        ]
    )
    
    return {
        "id": user_id,
        "name": name,
        "description": description,
        "img_url": img_url,
        "message": "User registered successfully"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)