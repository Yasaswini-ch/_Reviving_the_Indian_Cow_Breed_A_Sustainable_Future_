import io
import os
import logging
import base64
import traceback
import numpy as np
import cv2
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from roboflow import Roboflow
import supervision as sv

# Load environment variables
load_dotenv()
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Roboflow model
try:
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace().project("cattle-breed-9rfl6-xqimv-mqao3")
    model = project.version(6).model
except Exception as e:
    logger.error(f"Failed to initialize Roboflow model: {str(e)}")
    raise

@app.post("/predict/")
async def predict(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img_path = "temp.jpg"
        img.save(img_path)

        # Run inference
        result = model.predict(img_path, confidence=40, overlap=30).json()
        predictions = result.get("predictions", [])

        if not predictions:
            return JSONResponse(content={"error": "No objects detected"}, status_code=200)

        # Extract bounding box data
        labels, xyxys, confidences = [], [], []
        for item in predictions:
            x, y, width, height = item["x"], item["y"], item["width"], item["height"]
            confidence, label = item["confidence"], item["class"]

            x_min = x - (width / 2)
            y_min = y - (height / 2)
            x_max = x + (width / 2)
            y_max = y + (height / 2)

            xyxys.append([x_min, y_min, x_max, y_max])
            confidences.append(confidence)
            labels.append(f"{label} ({confidence * 100:.2f}%)")  # Adding confidence percentage

        # Convert to supervision detections format
        detections = sv.Detections(
            xyxy=np.array(xyxys),
            confidence=np.array(confidences),
            class_id=np.array([0] * len(labels))
        )

        # Read image for annotation
        image_cv2 = cv2.imread(img_path)
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        annotated_image = box_annotator.annotate(scene=image_cv2, detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

        # Convert to base64
        _, buffer = cv2.imencode(".png", annotated_image)
        img_str = base64.b64encode(buffer).decode()

        os.remove(img_path)

        return JSONResponse(content={
            "image": f"data:image/png;base64,{img_str}",
            "objects": labels  # Labels now include confidence scores
        })

    except Exception as e:
        logger.error(f"Error in object detection: {str(e)}\n{traceback.format_exc()}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)