from fastapi import FastAPI
import uvicorn
import ngrok

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello World"}

# Add your other API endpoints here

if __name__ == "__main__":
    # Start ngrok tunnel
    listener = ngrok.forward(addr="8000", authtoken_from_env=True)
    print(f"Ingress established at: {listener.url()}")
    
    # Start uvicorn server
    uvicorn.run(app, host="0.0.0.0", port=8000)