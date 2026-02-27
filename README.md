🌱 AI-Powered Smart Agriculture System (FastAPI + Deep Learning)

An end-to-end AI-driven agricultural assistant that performs:

- 🌿 Plant Disease Classification
- 🌍 Soil Type Detection
- 💧 Soil Moisture Prediction (Regression)
- 🌾 Weed Detection (YOLOv8 Object Detection)
- 📡 Real-Time Camera-Based Predictions via WebSockets

Built using FastAPI + TensorFlow/Keras + YOLOv8 + OpenCV for high-performance inference and real-time deployment.

---

🚀 Features

✅ Multi-model AI inference in a single backend
✅ Real-time prediction using WebSockets
✅ Image-based classification & detection
✅ Numerical regression for moisture forecasting
✅ Deployable as an API service
✅ Lightweight frontend served via FastAPI static files
✅ Designed for precision agriculture & smart farming

---

🧠 Models Used

Task| Model| Framework
Plant Disease Detection| CNN Classifier| TensorFlow/Keras
Soil Type Classification| MobileNetV2 Transfer Learning| TensorFlow
Soil Moisture Prediction| Regression Model| TensorFlow
Weed Detection| YOLOv8| Ultralytics
Real-Time Detection| OpenCV + WebSocket Streaming| FastAPI

---

📁 Project Structure

├── mainappp.py                # FastAPI Backend
├── requirements.txt           # Dependencies
├── static/                    # Frontend (HTML/CSS/JS)
├── leaf_disease_model.h5      # Disease classification model
├── soil_type_classifier.h5    # Soil classifier
├── soil_moisture.h5           # Moisture regression model
├── weed_detector.pt           # YOLOv8 weights
├── README.md

---

⚙️ Installation

1️⃣ Clone Repository

git clone https://github.com/NIS17/real-time-agricultural-field-analysis-and-detection.git
cd real-time-agricultural-field-analysis-and-detection

2️⃣ Create Virtual Environment

python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

3️⃣ Install Dependencies

pip install -r requirements.txt

---

▶️ Running the Server

python mainappp.py

Server starts at:

http://localhost:8000

---

📡 API Endpoints

🌿 Disease Prediction

POST /predict/disease

Upload leaf image → returns disease name + confidence.

---

🌍 Soil Type Classification

POST /predict/soil

Upload soil image → returns soil category.

---

💧 Soil Moisture Prediction

POST /predict/moisture

Input JSON:

{
  "Month": 5,
  "Day": 14,
  "avg_temp": 32.5,
  "avg_wind": 5.2,
  "avg_ws": 3.8,
  "avg_sol_rad": 210,
  "s_1_t": 24,
  "s_2_t": 25,
  "s_3_t": 26,
  "s_4_t": 27
}

Returns predicted moisture value.

---

🌾 Weed Detection

POST /predict/weed

Returns annotated image with bounding boxes.

---

📹 Real-Time Detection (WebSocket)

WS /ws/realtime

Supports:

- Live disease detection
- Live soil classification
- Live weed detection

---

🛠 Technologies Used

- FastAPI → High-performance backend
- TensorFlow/Keras → Deep learning inference
- YOLOv8 (Ultralytics) → Object detection
- OpenCV → Image processing
- WebSockets → Real-time streaming
- Pydantic → Data validation
- Uvicorn → ASGI server

---

🎯 Use Cases

✔ Smart farming automation
✔ Crop health monitoring
✔ Precision irrigation planning
✔ Autonomous drone agriculture
✔ Research in AI-based AgriTech
✔ Edge AI deployment for farms

---

📈 Future Improvements

- Add satellite imagery support
- Integrate IoT sensor pipeline
- Deploy using Docker + Kubernetes
- Add model retraining dashboard
- Edge deployment on Jetson/Raspberry Pi

---



📜 License

This project is for academic and research purposes.

