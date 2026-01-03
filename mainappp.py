from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from PIL import Image
import numpy as np
import io
import os
import cv2
import base64
from ultralytics import YOLO
from starlette.websockets import WebSocketState



# --- Pydantic model for moisture prediction input ---
class MoistureData(BaseModel):
    Month: int;
    Day: int;
    avg_temp: float;
    avg_wind: float;
    avg_ws: float
    avg_sol_rad: float;
    s_1_t: float;
    s_2_t: float;
    s_3_t: float;
    s_4_t: float


app = FastAPI()

# --- 1. Load ALL FOUR models and their configurations ---

# Plant Disease Model (Classifier)
DISEASE_MODEL_PATH = 'leaf_disease_model.h5'
disease_model = keras.models.load_model(DISEASE_MODEL_PATH)
DISEASE_CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]
print("✅ Plant Disease model loaded.")

# Soil Type Model (Classifier)
SOIL_MODEL_PATH = 'soil_type_classifier.h5'
soil_model = keras.models.load_model(SOIL_MODEL_PATH)
SOIL_CLASS_NAMES = ['Black Soil', 'Cinder Soil', 'Laterite Soil', 'Peat Soil', 'Yellow Soil']
print("✅ Soil Type model loaded.")

# Soil Moisture Model (Regressor)
MOISTURE_MODEL_PATH = 'soil_moisture.h5'
moisture_model = keras.models.load_model(MOISTURE_MODEL_PATH, compile=False)
print("✅ Soil Moisture model loaded.")

# Weed Detection Model (Object Detector)
WEED_MODEL_PATH = 'weed_detector.pt'  # Rename best.pt to this
weed_model = YOLO(WEED_MODEL_PATH)
print("✅ Weed Detection model loaded.")

# --- 2. Serve the Static Frontend ---
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("static/indexxxxx.html", "r") as f: return f.read()


# --- 3. Static Prediction Endpoints ---

@app.post("/predict/disease")
async def predict_disease(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert('RGB').resize((128, 128))
    img_array = np.array(image).astype('float32')
    img_array = np.expand_dims(img_array, axis=0)
    predictions = disease_model.predict(img_array, verbose=False)
    pred_index = np.argmax(predictions[0])
    return {"prediction": DISEASE_CLASS_NAMES[pred_index], "confidence": float(predictions[0][pred_index])}


@app.post("/predict/soil")
async def predict_soil(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert('RGB').resize((224, 224))
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = mobilenet_preprocess(img_array.astype('float32'))
    predictions = soil_model.predict(preprocessed_img, verbose=False)
    pred_index = np.argmax(predictions[0])
    return {"prediction": SOIL_CLASS_NAMES[pred_index], "confidence": float(predictions[0][pred_index])}


@app.post("/predict/moisture")
async def predict_moisture(data: MoistureData):
    input_data = np.array([[data.Month, data.Day, data.avg_temp, data.avg_wind, data.avg_ws, data.avg_sol_rad,
                            data.s_1_t, data.s_2_t, data.s_3_t, data.s_4_t]])
    prediction = moisture_model.predict(input_data, verbose=False)
    return {"predicted_moisture": float(prediction[0][0])}


@app.post("/predict/weed")
async def predict_weed(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    results = weed_model.predict(img_cv, verbose=False)
    annotated_frame = results[0].plot()
    _, buffer = cv2.imencode('.jpg', annotated_frame)
    encoded_image = base64.b64encode(buffer).decode('utf-8')
    return JSONResponse(content={"annotated_image": f"data:image/jpeg;base64,{encoded_image}"})


# --- 4. Real-Time WebSocket Endpoint ---
# --- 4. Real-Time WebSocket Endpoint ---
@app.websocket("/ws/realtime")
async def realtime_prediction(websocket: WebSocket):
    # Establish connection
    await websocket.accept()

    try:
        # Main loop for receiving/sending data
        while True:
            # 1. Receive data
            data = await websocket.receive_json()
            model_type = data['model_type']
            image_str = data['image_str'].split(';base64,')[-1]

            # 2. Decode image
            img_bytes = base64.b64decode(image_str)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            output_frame = frame

            # 3. Prediction Logic (DEFINES pred_name and confidence)
            if model_type == 'disease':
                # *** RESTORED LOGIC START ***
                img_resized = cv2.resize(frame, (128, 128))
                img_array = np.array(img_resized).astype('float32')
                img_array = np.expand_dims(img_array, axis=0)
                predictions = disease_model.predict(img_array, verbose=False)
                pred_index = np.argmax(predictions[0])
                confidence = predictions[0][pred_index] * 100
                pred_name = DISEASE_CLASS_NAMES[pred_index].replace('_', ' ')
                # *** RESTORED LOGIC END ***

                prediction_text = f"{pred_name} ({confidence:.1f}%)"
                cv2.putText(output_frame, prediction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            elif model_type == 'soil':
                # *** RESTORED LOGIC START ***
                img_resized = cv2.resize(frame, (224, 224))
                img_array = np.expand_dims(img_resized, axis=0)
                preprocessed_img = mobilenet_preprocess(img_array.astype('float32'))
                predictions = soil_model.predict(preprocessed_img, verbose=False)
                pred_index = np.argmax(predictions[0])
                confidence = predictions[0][pred_index] * 100
                pred_name = SOIL_CLASS_NAMES[pred_index]
                # *** RESTORED LOGIC END ***

                prediction_text = f"{pred_name} ({confidence:.1f}%)"
                cv2.putText(output_frame, prediction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            elif model_type == 'weed':
                results = weed_model.predict(frame, verbose=False)
                output_frame = results[0].plot()

            # 4. Encode and Send
            _, buffer = cv2.imencode('.jpg', output_frame)
            encoded_frame = base64.b64encode(buffer).decode('utf-8')
            await websocket.send_text(f"data:image/jpeg;base64,{encoded_frame}")

    except WebSocketDisconnect:
        # Expected exit when client closes connection. No explicit close() needed.
        print("Client disconnected gracefully.")

    except Exception as e:
        # Catches unexpected errors (e.g., JSON parse error)
        print(f"Error in WebSocket: {e}")

        # Only attempt to close if the connection isn't already closed or closing
        if websocket.application_state not in [WebSocketState.DISCONNECTED, WebSocketState.CLOSING]:
            await websocket.close()

    # --- 5. Run the Server ---
if __name__ == "__main__":
    import uvicorn
        # Make sure this line is present and correct:
    uvicorn.run("mainappp:app", host="0.0.0.0", port=8000, reload=True)