from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import torch
from utils import process_image, validate_image, load_model

app = FastAPI(title="NeuralFin")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained AI model
MODEL_PATH = "path/to/your/model.pth"  # Update with actual path
model = load_model(MODEL_PATH)

@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "NeuralFin API is online",
        "Project": "Shark Conservation AI"
    }

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    try:
        # Read the uploaded file
        contents = await file.read()

        # Validate image
        if not validate_image(contents):
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Process the image
        processed_image = process_image(contents)
        if processed_image is None:
            raise HTTPException(status_code=400, detail="Image processing failed")

        # Convert image to tensor for PyTorch
        image_tensor = torch.from_numpy(processed_image).unsqueeze(0)  # Add batch dimension

        # Perform inference
        with torch.no_grad():
            prediction = model(image_tensor)
            predicted_label = prediction.argmax(dim=1).item()

        return {
            "filename": file.filename,
            "status": "processed",
            "prediction": predicted_label,
            "message": "Image analyzed successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)