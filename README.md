# Face Recognition API

API for face recognition that returns identified people with bounding boxes and information.

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the API:
   ```
   uvicorn api:app --reload
   ```

The API will be available at http://localhost:8000

## API Endpoints

### POST /analyze

Upload an image for face recognition.

**Request:**
- Form data with a file named "file"

**Response:**
```json
[
  {
    "id": "uuid-string",
    "name": "Person Name",
    "thumbnailUrl": "base64-encoded-image",
    "description": "Confidence: 0.95",
    "faceBox": {
      "top": "10%",
      "left": "20%",
      "width": "15%",
      "height": "25%"
    }
  }
]
```

## Frontend Interface

Expected Profile interface:
```typescript
export interface Profile {
  id: string
  name: string
  thumbnailUrl: string
  description: string
  faceBox: { 
    top: string; 
    left: string; 
    width: string; 
    height: string 
  } // As percentage strings
}
```