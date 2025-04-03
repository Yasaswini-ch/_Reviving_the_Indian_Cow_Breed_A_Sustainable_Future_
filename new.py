import io
import os
import logging
import base64
import traceback
import numpy as np
import cv2
from PIL import Image
import uuid # Import for unique filenames
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from roboflow import Roboflow # Assuming RoboflowError isn't available
import supervision as sv
# import requests # If needed for direct API calls or specific error handling

# --- Load Environment Variables ---
load_dotenv() # Load .env file if present (useful for local testing)
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
ROBOFLOW_PROJECT_ID = os.getenv("ROBOFLOW_PROJECT_ID", "cattle-breed-9rfl6-xqimv-mqao3") # Allow override via env var
ROBOFLOW_MODEL_VERSION = int(os.getenv("ROBOFLOW_MODEL_VERSION", 6)) # Allow override via env var
CONFIDENCE_THRESHOLD = int(os.getenv("CONFIDENCE_THRESHOLD", 40))
OVERLAP_THRESHOLD = int(os.getenv("OVERLAP_THRESHOLD", 30))

# --- Initialize Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Validate API Key ---
if not ROBOFLOW_API_KEY:
    logger.error("FATAL: ROBOFLOW_API_KEY environment variable not set.")
    raise ValueError("Roboflow API Key is required but not configured.")

# --- Initialize Roboflow Model ---
model = None
try:
    logger.info("Initializing Roboflow...")
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    logger.info(f"Accessing workspace...")
    workspace = rf.workspace()
    logger.info(f"Accessing project: {ROBOFLOW_PROJECT_ID}")
    project = workspace.project(ROBOFLOW_PROJECT_ID)
    logger.info(f"Loading model version: {ROBOFLOW_MODEL_VERSION}")
    model = project.version(ROBOFLOW_MODEL_VERSION).model
    logger.info("Roboflow model loaded successfully.")
except Exception as e:
    error_message = str(e).lower()
    if "authenticate" in error_message or "api key" in error_message or "401" in error_message:
         logger.error(f"Roboflow Authentication/API Key Error during initialization: {e}")
    elif "project not found" in error_message or "404" in error_message or "workspace" in error_message:
         logger.error(f"Roboflow Project ('{ROBOFLOW_PROJECT_ID}') or Version ({ROBOFLOW_MODEL_VERSION}) not found, or Workspace access error: {e}")
    elif "connection" in error_message or "timeout" in error_message:
        logger.error(f"Network error connecting to Roboflow during initialization: {e}")
    else:
         logger.error(f"An unexpected error occurred during Roboflow initialization: {str(e)}")
    logger.error(traceback.format_exc())
    # Raise error to prevent server starting without model
    raise RuntimeError(f"Failed to initialize Roboflow model: {e}")

# --- Initialize FastAPI app ---
app = FastAPI(title="Cattle Breed Detection API")

# --- Enable CORS ---
# Allow requests from anywhere during development/testing.
# Restrict this in production if possible (e.g., to your Streamlit app's domain).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    """ Root endpoint for health check. """
    return {"message": "Cattle Breed Detection API is running."}

@app.post("/predict/")
async def predict(image: UploadFile = File(...)):
    """ Receives image, runs Roboflow prediction, returns annotated image and labels. """
    temp_image_path = None
    try:
        # 1. Read and Prepare Image
        logger.info(f"Received image: {image.filename} ({image.content_type})")
        contents = await image.read()
        if not contents:
             logger.warning("Received empty image file.")
             raise HTTPException(status_code=400, detail="Empty image file received.")

        try:
            pil_image = Image.open(io.BytesIO(contents)).convert("RGB")

            # Optional: Resize large images before prediction
            max_size = (1024, 1024) # Example max dimensions
            if pil_image.size[0] > max_size[0] or pil_image.size[1] > max_size[1]:
                 logger.info(f"Resizing image from {pil_image.size} to fit within {max_size}...")
                 pil_image.thumbnail(max_size, Image.Resampling.LANCZOS)
                 logger.info(f"Image resized to {pil_image.size}")

        except Exception as img_err:
             logger.error(f"Failed to open or process image: {img_err}")
             raise HTTPException(status_code=400, detail=f"Invalid or corrupted image file: {img_err}")

        image_cv2 = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        # Use UUID for unique temporary filename
        temp_image_path = f"temp_{uuid.uuid4()}.jpg"
        pil_image.save(temp_image_path)
        logger.info(f"Temporary image saved to: {temp_image_path}")

        # 2. Run Roboflow Inference
        logger.info(f"Running Roboflow prediction (Version: {ROBOFLOW_MODEL_VERSION}, Confidence: {CONFIDENCE_THRESHOLD}, Overlap: {OVERLAP_THRESHOLD})...")
        result = model.predict(temp_image_path, confidence=CONFIDENCE_THRESHOLD, overlap=OVERLAP_THRESHOLD).json()
        logger.info("Roboflow prediction completed.")

        # Check for errors reported within the Roboflow JSON response
        if "error" in result:
             error_msg = result.get("error", "Unknown Roboflow error in response")
             logger.error(f"Roboflow returned an error in the response: {error_msg}")
             status_code = 500
             # Try to determine a better status code
             if isinstance(error_msg, dict) and "message" in error_msg: error_str = error_msg["message"].lower()
             elif isinstance(error_msg, str): error_str = error_msg.lower()
             else: error_str = ""

             if "authenticate" in error_str: status_code = 500 # Internal config issue
             elif "limit" in error_str: status_code = 429 # Rate limit
             else: status_code = 503 # Generic service issue
             raise HTTPException(status_code=status_code, detail=f"Roboflow prediction failed: {error_msg}")

        predictions = result.get("predictions", [])
        logger.info(f"Found {len(predictions)} predictions.")

        if not predictions:
            logger.info("No objects detected.")
            return JSONResponse(content={"image": None, "objects": []}, status_code=200)

        # 3. Process Predictions
        labels, xyxys, confidences = [], [], []
        detected_classes = set()
        for item in predictions:
            x_center, y_center, width, height = item["x"], item["y"], item["width"], item["height"]
            confidence, label = item["confidence"], item["class"]
            detected_classes.add(label)
            x_min = x_center - (width / 2); y_min = y_center - (height / 2)
            x_max = x_center + (width / 2); y_max = y_center + (height / 2)
            xyxys.append([x_min, y_min, x_max, y_max])
            confidences.append(confidence)
            labels.append(f"{label} ({confidence * 100:.1f}%)")

        logger.info(f"Detected classes: {list(detected_classes)}")
        detections = sv.Detections(
            xyxy=np.array(xyxys),
            confidence=np.array(confidences),
            class_id=np.array(range(len(labels)))
        )

        # 4. Annotate Image
        # Use corrected BoxAnnotator initialization
        box_annotator = sv.BoxAnnotator(thickness=2)
        annotated_image = box_annotator.annotate(
            scene=image_cv2.copy(),
            detections=detections,
            labels=labels
            )
        logger.info("Image annotation completed.")

        # 5. Encode Annotated Image
        is_success, buffer = cv2.imencode(".png", annotated_image)
        if not is_success:
             logger.error("Failed to encode annotated image to PNG buffer.")
             raise HTTPException(status_code=500, detail="Failed to encode result image.")
        img_str = base64.b64encode(buffer).decode("utf-8")
        base64_image = f"data:image/png;base64,{img_str}"
        logger.info("Annotated image encoded to base64.")

        # 6. Return Response
        return JSONResponse(content={
            "image": base64_image,
            "objects": labels
        })

    except HTTPException as http_exc:
         # Re-raise known HTTP exceptions
         raise http_exc
    except Exception as e:
        # Catch-all for unexpected errors during prediction/processing
        logger.error(f"An unexpected error occurred in /predict/: {str(e)}")
        logger.error(traceback.format_exc())
        # Determine if it's a connection error to provide a 503 status
        error_message = str(e).lower()
        status_code = 500
        detail_message = "Prediction failed: An internal server error occurred."
        if "connection" in error_message or "timeout" in error_message or "remote disconnected" in error_message:
            status_code = 503
            detail_message = "Prediction failed: Could not connect to Roboflow API or connection lost."

        return JSONResponse(content={"error": detail_message}, status_code=status_code)
    finally:
        # Ensure temporary file is always deleted
        if temp_image_path and os.path.exists(temp_image_path):
            try:
                os.remove(temp_image_path)
                logger.info(f"Temporary image file deleted: {temp_image_path}")
            except Exception as del_err:
                logger.error(f"Error deleting temporary file {temp_image_path}: {del_err}")

# --- Run the app (for Render/PaaS) ---
if __name__ == "__main__":
    import uvicorn
    # Read port from environment variable provided by Render/Heroku etc.
    # Default to 8000 for local testing if PORT isn't set.
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting Cattle Breed Detection API server on host 0.0.0.0:{port}")
    # Use "0.0.0.0" to listen on all interfaces within the container
    # The application object is specified as "new:app" (filename:variable)
    uvicorn.run("new:app", host="0.0.0.0", port=port)
    # DO NOT use reload=True in production deployments
