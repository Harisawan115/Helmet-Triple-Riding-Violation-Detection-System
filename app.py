import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import tempfile
import os
import time
from datetime import datetime
import pandas as pd
from collections import deque
from ultralytics import YOLO
import easyocr
from dataclasses import dataclass
import io

# Page config
st.set_page_config(
    page_title="Helmet & Triple Riding Detection",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #FF4B4B, #6C63FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
    }
    .violation-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        margin: 8px 0;
        border-left: 5px solid #FFD700;
    }
    .error-box {
        background: #ffebee;
        border: 2px solid #f44336;
        padding: 10px;
        border-radius: 5px;
        color: #c62828;
    }
    .success-box {
        background: #e8f5e9;
        border: 2px solid #4caf50;
        padding: 10px;
        border-radius: 5px;
        color: #2e7d32;
    }
</style>
""", unsafe_allow_html=True)

@dataclass
class VehicleTrack:
    id: int
    bbox_history: deque
    center_history: deque
    violation_confirmed: bool
    license_plate: str
    frames_since_seen: int
    class_type: str

class ResNetHelmetClassifier(nn.Module):
    def __init__(self, num_classes=2, dropout=0.5):
        super(ResNetHelmetClassifier, self).__init__()
        self.resnet = models.resnet18(pretrained=False)
        # Freeze only first block - matches train_models.py exactly
        for name, param in self.resnet.named_parameters():
            if 'layer1' in name or 'conv1' in name or 'bn1' in name:
                param.requires_grad = False
        num_features = self.resnet.fc.in_features
        # MUST match train_models.py FC head exactly:
        # Linear -> BatchNorm1d -> ReLU -> Dropout -> Linear
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.resnet(x)

class TrafficViolationSystem:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.yolo = YOLO("yolov8n.pt") 
        self.frame_count = 0
        self.tracks = {}
        self.next_track_id = 0
        
        # Load models
        self.load_yolo()
        self.load_helmet_classifier()
        self.init_ocr()
        
        # Transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        
    @st.cache_resource
    def load_yolo(_self):
        with st.spinner('🔄 Loading YOLOv8...'):
            model = YOLO('yolov8n.pt')
            model.to(_self.device)
        return model
    
    def load_helmet_classifier(self):
        model_path = 'models/helmet_resnet18.pth'
        if os.path.exists(model_path):
            try:
                self.helmet_model = ResNetHelmetClassifier()
                checkpoint = torch.load(model_path, map_location=self.device)
                self.helmet_model.load_state_dict(checkpoint)
                self.helmet_model.to(self.device)
                self.helmet_model.eval()
                st.success("✅ ResNet18 Loaded")
            except Exception as e:
                st.warning(f"⚠️ ResNet load failed: {e}")
                self.helmet_model = None
        else:
            st.info("ℹ️ Using fallback helmet detection")
            self.helmet_model = None
    
    def init_ocr(self):
        with st.spinner('🔄 Loading OCR...'):
            try:
                self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available(), verbose=False)
                st.success("✅ OCR Ready")
            except Exception as e:
                st.error(f"OCR failed: {e}")
                self.reader = None
    
    def get_iou(self, box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def update_tracks(self, detections):
        new_tracks = {}
        
        for det in detections:
            x1, y1, x2, y2, cls_id, conf, cls_name = det
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            bbox = (x1, y1, x2, y2)
            
            best_match = None
            best_iou = 0.3
            
            for tid, track in self.tracks.items():
                if track.class_type == cls_name and track.frames_since_seen < 5:
                    last_bbox = track.bbox_history[-1]
                    iou = self.get_iou(bbox, last_bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_match = tid
            
            if best_match is not None:
                track = self.tracks[best_match]
                track.bbox_history.append(bbox)
                track.center_history.append(center)
                track.frames_since_seen = 0
                new_tracks[best_match] = track
            else:
                new_track = VehicleTrack(
                    id=self.next_track_id,
                    bbox_history=deque([bbox], maxlen=30),
                    center_history=deque([center], maxlen=30),
                    violation_confirmed=False,
                    license_plate="",
                    frames_since_seen=0,
                    class_type=cls_name,
                )
                new_tracks[self.next_track_id] = new_track
                self.next_track_id += 1
        
        for tid in self.tracks:
            if tid not in new_tracks:
                self.tracks[tid].frames_since_seen += 1
        
        self.tracks = new_tracks
        return self.tracks
    
    def classify_helmet(self, head_region):
        if head_region.size < 1000:
            return True, 0.5
        
        if self.helmet_model is not None:
            try:
                head_rgb = cv2.cvtColor(head_region, cv2.COLOR_BGR2RGB)
                head_pil = Image.fromarray(head_rgb)
                input_tensor = self.transform(head_pil).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    outputs = self.helmet_model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    helmet_prob = probabilities[0][1].item()
                
                return helmet_prob > 0.5, helmet_prob
            except:
                pass
        
        # Fallback
        hsv = cv2.cvtColor(head_region, cv2.COLOR_BGR2HSV)
        colors = [((0,50,50),(10,255,255)), ((160,50,50),(180,255,255)),
                  ((100,50,50),(130,255,255)), ((20,100,100),(40,255,255))]
        
        total = head_region.shape[0] * head_region.shape[1]
        helmet_pixels = sum(cv2.countNonZero(cv2.inRange(hsv, np.array(l), np.array(u))) 
                           for l,u in colors)
        score = helmet_pixels / total
        return score > 0.3, score
    
    def read_license_plate(self, vehicle_roi):
        if self.reader is None:
            return "NO_OCR", 0.0
        
        try:
            gray = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2GRAY)
            results = self.reader.readtext(gray)
            
            if results:
                best = max(results, key=lambda x: x[2])
                plate_text = best[1].upper().replace(" ", "")
                confidence = best[2]
                if plate_text.isalnum() and 5 <= len(plate_text) <= 10:
                    return plate_text, confidence
            return "UNKNOWN", 0.0
        except:
            return "ERROR", 0.0
    
    def process_frame(self, frame, debug_mode=False):
        self.frame_count += 1
        h, w = frame.shape[:2]
        
        # YOLO Detection
        results = self.yolo(frame, verbose=False, conf=0.3)  # Lower confidence for testing
        detections = []
        
        for det in results[0].boxes:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            cls_id = int(det.cls)
            conf = float(det.conf)
            cls_name = self.yolo.names[cls_id]
            
            if cls_name in ['person', 'car', 'motorcycle', 'bus', 'truck']:
                detections.append((x1, y1, x2, y2, cls_id, conf, cls_name))
        
        # Update tracks
        tracks = self.update_tracks(detections)
        
        # Detect violations
        violations = []
        debug_info = []
        
        # Match motorcycles to their tracks
        track_list = list(tracks.values())
        det_track_pairs = []
        for det in detections:
            bbox = (det[0], det[1], det[2], det[3])
            for track in track_list:
                if track.class_type == det[6] and track.bbox_history[-1] == bbox:
                    det_track_pairs.append((det, track))
                    break

        motorcycles = [(d, t) for d, t in det_track_pairs if d[6] == 'motorcycle']
        persons     = [d for d in detections if d[6] == 'person']

        if debug_mode:
            debug_info.append(f"Frame: {self.frame_count}")
            debug_info.append(f"Motorcycles: {len(motorcycles)}, Persons: {len(persons)}")

        # Helmet & Triple Riding
        for det, track in motorcycles:
            mx1, my1, mx2, my2, _, _, _ = det
            riders = []
            
            for person in persons:
                px1, py1, px2, py2, _, _, _ = person
                p_center_x = (px1 + px2) // 2
                p_center_y = (py1 + py2) // 2
                
                if mx1 < p_center_x < mx2 and my1 < p_center_y < my2:
                    riders.append(person)
                    
                    head_y1 = max(0, py1)
                    head_y2 = min(h, py1 + (py2 - py1) // 3)
                    head_region = frame[head_y1:head_y2, px1:px2]
                    
                    has_helmet, helmet_conf = self.classify_helmet(head_region)
                    
                    if not has_helmet:
                        try:
                            vehicle_roi = frame[my1:my2, mx1:mx2]
                            plate, plate_conf = self.read_license_plate(vehicle_roi)
                        except:
                            plate, plate_conf = "N/A", 0.0
                        
                        violations.append({
                            'track_id': track.id,
                            'type': '🪖 No Helmet',
                            'bbox': (px1, py1, px2, py2),
                            'license_plate': plate,
                            'helmet_confidence': helmet_conf,
                            'timestamp': datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        })
            
            if len(riders) > 2:
                try:
                    vehicle_roi = frame[my1:my2, mx1:mx2]
                    plate, plate_conf = self.read_license_plate(vehicle_roi)
                except:
                    plate, plate_conf = "N/A", 0.0
                
                violations.append({
                    'track_id': track.id,
                    'type': '👥 Triple Riding',
                    'bbox': (mx1, my1, mx2, my2),
                    'rider_count': len(riders),
                    'license_plate': plate,
                    'timestamp': datetime.now().strftime("%H:%M:%S.%f")[:-3]
                })
        
        # Draw
        annotated = self.draw_annotations(frame.copy(), violations, tracks, debug_info if debug_mode else [])
        return annotated, violations, debug_info
    
    def draw_annotations(self, frame, violations, tracks, debug_info):
        h, w = frame.shape[:2]
        
        # Tracks
        for tid, track in tracks.items():
            if track.bbox_history:
                x1, y1, x2, y2 = track.bbox_history[-1]
                color = (255, 255, 0) if track.class_type == 'car' else (0, 255, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID:{track.id}", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
        
        # Violations
        for v in violations:
            x1, y1, x2, y2 = v['bbox']
            
            if 'Helmet' in v['type']:
                color = (0, 140, 255)
                thickness = 3
            else:
                color = (255, 0, 255)
                thickness = 3
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            plate = v.get('license_plate', 'N/A')
            label = f"{v['type']} | {plate}"
            
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Stats
        cv2.rectangle(frame, (10, h - 120), (500, h - 10), (0, 0, 0), -1)
        cv2.putText(frame, f"Violations: {len(violations)}", (20, h - 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Frame: {self.frame_count}", (20, h - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Debug info
        if debug_info:
            y_offset = 200
            for info in debug_info[:15]:
                cv2.putText(frame, str(info)[:80], (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                y_offset += 15
        
        return frame

def main():
    st.markdown('<h1 class="main-header">🪖 Helmet & Triple Riding Detection</h1>', 
                unsafe_allow_html=True)
    
    # Initialize
    if 'system' not in st.session_state:
        st.session_state.system = TrafficViolationSystem()
        st.session_state.violations = []
        st.session_state.error_log = []
    
    system = st.session_state.system
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        input_mode = st.radio("Input Mode:", ["📁 Upload Image/Video", "📷 Webcam"])
        debug_mode = st.checkbox("🔍 Debug Mode", help="Show detection details")
        
        st.markdown("---")
        st.header("🧠 Status")
        st.info(f"Device: {system.device}")
        st.info(f"YOLOv8: ✅")
        st.info(f"ResNet18: {'✅' if system.helmet_model else '⚠️'}")
        st.info(f"OCR: {'✅' if system.reader else '❌'}")
        
        st.markdown("---")
        
        if st.button("🗑️ Clear Violations"):
            st.session_state.violations = []
            st.experimental_rerun()
        
        if st.session_state.violations:
            df = pd.DataFrame(st.session_state.violations)
            csv = df.to_csv(index=False)
            st.download_button("📥 Export CSV", csv, "violations.csv", "text/csv")
    
    # Main content
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("📹 Detection Feed")
        video_placeholder = st.empty()
        
        if debug_mode:
            st.subheader("🔍 Debug Info")
            debug_placeholder = st.empty()
    
    with col2:
        st.subheader("📊 Statistics")
        stats_placeholder = st.empty()
        
        st.subheader("🚨 Recent Violations")
        log_placeholder = st.empty()
        
        if st.session_state.error_log:
            st.subheader("⚠️ Errors")
            for err in st.session_state.error_log[-3:]:
                st.markdown(f'<div class="error-box">{err}</div>', unsafe_allow_html=True)
    
    # Processing
    if input_mode == "📁 Upload Image/Video":
        uploaded = st.file_uploader("Upload file", type=['jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov', 'webp'])
        
        if uploaded:
            try:
                # FIX: Proper image loading
                file_bytes = uploaded.read()
                
                # Try to load as image first
                try:
                    # Method 1: PIL
                    image = Image.open(io.BytesIO(file_bytes))
                    image = image.convert('RGB')
                    frame = np.array(image)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
                    is_video = False
                    st.success(f"✅ Image loaded: {image.size}")
                    
                except Exception as img_err:
                    # Method 2: Try as video
                    st.info("Not an image, trying video...")
                    is_video = True
                    
                    # Save to temp file for video
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    tfile.write(file_bytes)
                    tfile.close()
                    
                    cap = cv2.VideoCapture(tfile.name)
                    if not cap.isOpened():
                        raise Exception("Could not open video file")
                
                if not is_video:
                    # Process single image
                    with st.spinner('🔍 Analyzing...'):
                        annotated, violations, debug_info = system.process_frame(frame, debug_mode)
                    
                    video_placeholder.image(annotated, channels="BGR", use_column_width=True)
                    st.session_state.violations.extend(violations)
                    
                    if debug_mode:
                        debug_placeholder.code("\\n".join(debug_info))
                    
                    update_stats(stats_placeholder, log_placeholder)
                    
                    # Show success message
                    if violations:
                        st.markdown(f'<div class="success-box">✅ {len(violations)} violation(s) detected!</div>', 
                                  unsafe_allow_html=True)
                    else:
                        st.info("No violations detected in this frame")
                
                else:
                    # Process video
                    stframe = st.empty()
                    frame_count = 0
                    
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # Convert BGR to RGB for processing
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        annotated, violations, debug_info = system.process_frame(frame_rgb, debug_mode)
                        
                        # Convert back to BGR for display
                        annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
                        
                        stframe.image(annotated_bgr, channels="BGR", use_column_width=True)
                        st.session_state.violations.extend(violations)
                        
                        if debug_mode:
                            debug_placeholder.code("\\n".join(debug_info))
                        
                        update_stats(stats_placeholder, log_placeholder)
                        
                        frame_count += 1
                        if frame_count > 300:  # Limit to 10 seconds at 30fps
                            st.warning("Video truncated to first 10 seconds")
                            break
                    
                    cap.release()
                    st.success(f"✅ Processed {frame_count} frames")
                    
            except Exception as e:
                error_msg = f"Error processing file: {str(e)}"
                st.session_state.error_log.append(error_msg)
                st.error(error_msg)
                st.info("💡 Try: Convert image to JPG format or check if file is corrupted")
    
    else:  # Webcam
        run_webcam(system, video_placeholder, stats_placeholder, log_placeholder, 
                  debug_placeholder if debug_mode else None, debug_mode)

def run_webcam(system, video_placeholder, stats_placeholder, log_placeholder, debug_placeholder, debug_mode):
    """Run webcam with error handling"""
    st.info("Click 'Start' to begin webcam")
    
    col1, col2 = st.columns(2)
    with col1:
        start = st.button("▶️ Start Webcam", use_container_width=True)
    with col2:
        stop = st.button("⏹️ Stop", use_container_width=True)
    
    if start:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Could not open webcam. Check permissions.")
            return
        
        st.session_state.cam_running = True
        
        while st.session_state.get('cam_running', False) and not stop:
            try:
                ret, frame = cap.read()
                if not ret:
                    st.error("Camera read failed")
                    break
                
                # BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                annotated, violations, debug_info = system.process_frame(frame_rgb, debug_mode)
                
                # RGB to BGR for display
                annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
                
                video_placeholder.image(annotated_bgr, channels="BGR", use_column_width=True)
                st.session_state.violations.extend(violations)
                
                if debug_mode and debug_placeholder:
                    debug_placeholder.code("\\n".join(debug_info))
                
                update_stats(stats_placeholder, log_placeholder)
                
                # Limit history
                if len(st.session_state.violations) > 100:
                    st.session_state.violations = st.session_state.violations[-100:]
                    
            except Exception as e:
                st.error(f"Frame processing error: {e}")
                break
        
        cap.release()
        st.session_state.cam_running = False

def update_stats(stats_placeholder, log_placeholder):
    """Update statistics display"""
    violations = st.session_state.violations
    
    counts = {}
    for v in violations:
        counts[v['type']] = counts.get(v['type'], 0) + 1
    
    with stats_placeholder.container():
        cols = st.columns(2)
        cols[0].metric("🪖 No Helmet", counts.get('🪖 No Helmet', 0))
        cols[1].metric("👥 Triple Riding", counts.get('👥 Triple Riding', 0))
    
    with log_placeholder.container():
        recent = violations[-5:][::-1]
        for v in recent:
            plate = v.get('license_plate', 'N/A')
            debug = v.get('debug', '')
            st.markdown(f"""
            <div class="violation-card">
                <b>{v['type']}</b><br>
                <small>🚗 {plate} | ⏰ {v['timestamp']}</small>
                {f"<br><small style='color:#FFD700'>{debug}</small>" if debug else ""}
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()