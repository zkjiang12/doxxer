# API Documentation

## Overview
This API provides face recognition services through a FastAPI backend. The API can be accessed via the ngrok URL that's generated when running the server.

## API Endpoints

### 1. Analyze Image (`POST /analyze`)

Analyzes an uploaded image to detect and identify faces.

#### Request
- **Method:** POST
- **Content-Type:** multipart/form-data
- **Body Parameters:**
  - `file`: The image file to analyze (required)

#### Response
- **Content-Type:** application/json
- **Format:** Array of Profile objects
- **Profile Object:**
  ```json
  {
    "id": "string",          // Unique identifier for this detection
    "name": "string",        // Name of the identified person
    "thumbnailUrl": "string", // Base64 encoded thumbnail of the face
    "description": "string",  // Description (includes confidence score)
    "faceBox": {              // Face bounding box in percentage values
      "top": "string",        // e.g. "10.5%"
      "left": "string",       // e.g. "20.3%"
      "width": "string",      // e.g. "15.2%"
      "height": "string"      // e.g. "22.8%"
    }
  }
  ```

## Frontend Integration

### Example Usage (JavaScript)

```javascript
// Function to upload and analyze an image
async function analyzeImage(imageFile) {
  const formData = new FormData();
  formData.append('file', imageFile);
  
  try {
    const response = await fetch('https://your-ngrok-url/analyze', {
      method: 'POST',
      body: formData,
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }
    
    const profiles = await response.json();
    return profiles;
  } catch (error) {
    console.error('Error analyzing image:', error);
    throw error;
  }
}

// Example usage in a React component
function ImageUploader() {
  const [profiles, setProfiles] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  
  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    
    setIsLoading(true);
    try {
      const results = await analyzeImage(file);
      setProfiles(results);
    } catch (error) {
      console.error('Failed to analyze image:', error);
    } finally {
      setIsLoading(false);
    }
  };
  
  return (
    <div>
      <input type="file" accept="image/*" onChange={handleFileUpload} />
      {isLoading && <p>Analyzing image...</p>}
      
      {profiles.map(profile => (
        <div key={profile.id} style={{
          position: 'relative',
          border: '2px solid red',
          padding: '10px',
          margin: '10px 0'
        }}>
          <h3>{profile.name}</h3>
          <p>{profile.description}</p>
          <img src={profile.thumbnailUrl} alt={profile.name} width="100" />
        </div>
      ))}
    </div>
  );
}
```

### Displaying Face Boxes on Original Image

To display face boxes on the original image:

```javascript
function ImageWithFaceBoxes({ imageUrl, profiles }) {
  return (
    <div style={{ position: 'relative' }}>
      <img 
        src={imageUrl} 
        style={{ width: '100%', height: 'auto' }} 
        alt="Analyzed image" 
      />
      
      {profiles.map(profile => (
        <div 
          key={profile.id}
          style={{
            position: 'absolute',
            top: profile.faceBox.top,
            left: profile.faceBox.left,
            width: profile.faceBox.width,
            height: profile.faceBox.height,
            border: '2px solid red',
            boxSizing: 'border-box',
          }}
        >
          <div style={{
            background: 'rgba(255,0,0,0.7)',
            color: 'white',
            padding: '2px 5px',
            fontSize: '12px',
            position: 'absolute',
            bottom: '0',
            left: '0',
            right: '0'
          }}>
            {profile.name}
          </div>
        </div>
      ))}
    </div>
  );
}
```