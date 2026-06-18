# 🪖 Helmet & Triple Riding Violation Detection System

An AI-powered traffic violation detection system that automatically identifies **helmet non-compliance** and **triple riding** on motorcycles from images, video files, or a live webcam feed — built with YOLOv8n, ResNet18, and Streamlit.

---
## 🚀 Demo

| Feature | Description |
|---------|-------------|
| 🪖 No Helmet Detection | Detects motorcycle riders not wearing a helmet |
| 👥 Triple Riding | Detects 3 or more persons on a single motorcycle |
| 📷 Input Modes | Image upload, Video upload, Live Webcam |
| 📊 Violation Log | Real-time log with timestamps, exportable to CSV |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Object Detection | [YOLOv8n](https://github.com/ultralytics/ultralytics) — COCO pre-trained |
| Helmet Classifier | ResNet18 (PyTorch) — fine-tuned |
| Vehicle Tracker | Custom IoU-based tracker |
| Web Interface | [Streamlit](https://streamlit.io) |
| Image Processing | OpenCV, Pillow |
| Data Handling | Pandas, NumPy |

---

## 📦 Dataset

### Helmet Detection Dataset
The ResNet18 classifier is trained on a motorcycle helmet detection dataset containing real images of motorcycle riders with and without helmets.

**Download from Roboflow Universe:**

👉 **[Helmet Detection Dataset — Roboflow Universe](https://universe.roboflow.com/leo-ueno/helmet-detection-for-motorcyclists)**

Alternative dataset:

👉 **[Bike Helmet Detection — Roboflow](https://universe.roboflow.com/bike-helmets/bike-helmet-detection-2vdjo)**

**After downloading:**
1. Select **Folder Structure** format when downloading
2. Place the images in the following structure:

```
data/
└── helmet_dataset/
    ├── with_helmet/       ← images of riders wearing helmets
    └── without_helmet/    ← images of riders without helmets
```

> **Note:** YOLOv8n uses COCO pre-trained weights and downloads automatically — no dataset needed for detection.

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/helmet-triple-riding-detection.git
cd helmet-triple-riding-detection
```

### 2. Create a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the dataset
Download from the link above and place images in `data/helmet_dataset/with_helmet/` and `data/helmet_dataset/without_helmet/`.

---

## 🏋️ Training the Models

Run the training script to train ResNet18 and the LSTM trajectory predictor:

```bash
python train_models.py
```

**What gets trained:**

| Model | Epochs | Output File |
|-------|--------|-------------|
| ResNet18 Helmet Classifier | 25 | `models/helmet_resnet18.pth` |
| LSTM Trajectory Predictor | 50 | `models/lstm_tracker.pth` |

> YOLOv8n (`yolov8n.pt`) downloads automatically on first run — no training needed.

**Training output files saved to `outputs/`:**
- `resnet_training_curves.png` — loss and accuracy curves
- `confusion_matrix.png` — validation confusion matrix

---

## ▶️ Running the Application

```bash
streamlit run app.py
```

Then open your browser at `http://localhost:8501`

**Sidebar options:**
- Confidence threshold (default: 0.3)
- Debug mode toggle

---

## 🧠 How It Works

### Violation Detection Pipeline

```
Video Frame
    │
    ▼
YOLOv8n ──────────────────── Detects motorcycles + persons
    │
    ▼
IoU Tracker ──────────────── Assigns consistent IDs across frames
    │
    ├──► Helmet Check
    │         │
    │    Crop head region (top 1/3 of rider)
    │         │
    │    ResNet18 classifier
    │         │
    │    helmet_prob < 0.5 → 🪖 NO HELMET VIOLATION
    │
    └──► Triple Riding Check
              │
         Count persons inside motorcycle bbox
              │
         count ≥ 3 → 👥 TRIPLE RIDING VIOLATION
```

### Models Explained

**YOLOv8n** — detects all motorcycles and persons in the frame. Uses COCO pre-trained weights. No training required.

**ResNet18** — binary image classifier fine-tuned to predict `with_helmet` vs `without_helmet`. Uses transfer learning from ImageNet weights. Custom FC head: `Linear(512→256) → BatchNorm → ReLU → Dropout(0.5) → Linear(256→2)`.

**IoU Tracker** — tracks each motorcycle across frames using Intersection over Union matching (threshold: 0.3). Prevents duplicate violation counts for the same vehicle.

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Dataset Size | 574 images |
| With Helmet | 252 images |
| Without Helmet | 322 images |
| Train / Val Split | 80% / 20% |
| Best Validation Accuracy | 56.52% |
| Training Epochs | 25 |

> Accuracy can be improved significantly by using a larger dataset (3000+ images per class).
---

## 📌 Notes

- The system runs on **CPU** by default. If a CUDA-compatible GPU is available, it will be used automatically.
- Video processing is capped at **300 frames** (~10 seconds at 30fps) in video upload mode.
- If `helmet_resnet18.pth` is missing, the app falls back to a basic color-based helmet detector with lower accuracy.
- Run `train_models.py` before `app.py` to generate the model weights.
