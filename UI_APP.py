import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
import math
import os
import tempfile
import time

# --- CONFIGURATION ---
MODEL_PATH = "best_model_cnn_vit_L6.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Hyperparameters (must match the trained model)
SEQ_LENGTH = 30
VIT_EMBED_DIM = 768
VIT_NUM_HEADS = 12
VIT_NUM_LAYERS = 6
VIT_MLP_DIM = 3072
FLOW_DIM = 2
NUM_CLASSES = 2
DROPOUT_RATE = 0.5
OVERLAP = 15

IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- MODEL DEFINITION (Copied from your script) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class DeepfakeDetectionModel_CNNViT(nn.Module):
    def __init__(self):
        super(DeepfakeDetectionModel_CNNViT, self).__init__()
        _resnet = models.resnet18(weights=None)
        self.cnn_backbone = nn.Sequential(*list(_resnet.children())[:-1])
        self.cnn_feature_proj = nn.Linear(512, VIT_EMBED_DIM)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, VIT_EMBED_DIM))
        self.pos_encoder = PositionalEncoding(VIT_EMBED_DIM, max_len=SEQ_LENGTH + 1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=VIT_EMBED_DIM, nhead=VIT_NUM_HEADS, dim_feedforward=VIT_MLP_DIM,
            dropout=DROPOUT_RATE, batch_first=True, activation='gelu'
        )
        self.vit_encoder = nn.TransformerEncoder(encoder_layer, num_layers=VIT_NUM_LAYERS)
        self.classifier = nn.Sequential(
            nn.LayerNorm(VIT_EMBED_DIM + FLOW_DIM),
            nn.Linear(VIT_EMBED_DIM + FLOW_DIM, VIT_EMBED_DIM // 2), nn.GELU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(VIT_EMBED_DIM // 2, NUM_CLASSES)
        )

    def forward(self, frames, flow_features_sequence=None):
        batch_size, seq_len, _, _, _ = frames.size()
        cnn_features_list = [self.cnn_backbone(frames[:, t, :, :, :]).view(batch_size, -1) for t in range(seq_len)]
        cnn_sequence_features = torch.stack(cnn_features_list, dim=1)
        vit_input_features = self.cnn_feature_proj(cnn_sequence_features)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, vit_input_features), dim=1)
        x = self.pos_encoder(x)
        vit_encoded_output = self.vit_encoder(x)
        vit_sequence_output = vit_encoded_output[:, 0]
        if flow_features_sequence is not None:
            aggregated_flow_features = torch.mean(flow_features_sequence, dim=1)
        else:
            aggregated_flow_features = torch.zeros(batch_size, FLOW_DIM, device=frames.device)
        combined_features = torch.cat((vit_sequence_output, aggregated_flow_features), dim=1)
        return self.classifier(combined_features)

# --- INFERENCE HELPER FUNCTIONS ---

# Use Streamlit's caching to load the model only once
@st.cache_resource
def load_model():
    model = DeepfakeDetectionModel_CNNViT().to(DEVICE)
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}. Please ensure the model is in the 'models' directory.")
        return None
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

def compute_optical_flow(frames):
    flows = [np.array([0.0, 0.0], dtype=np.float32)]
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    for i in range(1, len(frames)):
        next_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        flows.append(np.array([np.mean(mag), np.var(mag)], dtype=np.float32))
        prev_gray = next_gray
    return torch.tensor(np.array(flows), dtype=torch.float32).unsqueeze(0)

def frames_to_tensor(frames_seq):
    imgs = [IMG_TRANSFORM(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))) for f in frames_seq]
    return torch.stack(imgs).unsqueeze(0)

# Modified predict_video to yield updates for the UI
def predict_video(video_path, model, max_sequences=5):
    cap = cv2.VideoCapture(video_path)
    frames = [f for ret, f in iter(lambda: cap.read(), (False, None))]
    cap.release()

    if not frames:
        st.error("Error: Could not read any frames from the video.")
        return

    if len(frames) < SEQ_LENGTH:
        frames.extend([frames[-1]] * (SEQ_LENGTH - len(frames)))

    sequences = []
    step = SEQ_LENGTH - OVERLAP
    for start in range(0, len(frames) - SEQ_LENGTH + 1, step):
        sequences.append(frames[start : start + SEQ_LENGTH])
        if len(sequences) >= max_sequences:
            break
            
    if not sequences:
        st.warning("Video is too short to create a full sequence for analysis.")
        return

    seq_preds = []
    with torch.no_grad():
        for i, seq in enumerate(sequences):
            yield {"status": "Processing", "progress": (i + 1) / len(sequences), "sequence": i + 1, "total": len(sequences)}
            
            frames_tensor = frames_to_tensor(seq).to(DEVICE)
            flow_tensor = compute_optical_flow(seq).to(DEVICE)
            out = model(frames_tensor, flow_tensor)
            probs = torch.softmax(out, dim=1).squeeze().cpu().numpy()
            prediction = int(np.argmax(probs))
            seq_preds.append(prediction)

    final_pred = max(set(seq_preds), key=seq_preds.count)
    yield {"status": "Complete", "final_prediction": final_pred, "sequence_predictions": seq_preds}

# --- STREAMLIT UI ---

st.set_page_config(layout="wide", page_title="AI Video Detector", page_icon="🤖")

# --- Sidebar ---
with st.sidebar:
    st.title("🤖 AI Video Detector")
    st.markdown("---")
    st.info("This application uses a hybrid CNN-Vision Transformer (ViT) architecture to detect fully AI-generated videos.")
    
    st.header("Upload Video")
    uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "mov", "avi", "mkv"])
    
    st.markdown("---")
    st.markdown("**Model:** `CNN-ViT`")
    st.markdown("**Accuracy (Test Set):** `99.42%`")
    st.markdown("**Accuracy (Unseen Data):** `91.60%`")
    st.markdown("---")
    st.write("Developed by Tan ZenYi")


# --- Main Content ---
st.title("AI-Generated Video Detection")
st.markdown("Upload a video via the sidebar to begin analysis.")

if uploaded_file is not None:
    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{uploaded_file.name.split(".")[-1]}') as tfile:
        tfile.write(uploaded_file.read())
        temp_video_path = tfile.name

    # Display the uploaded video in a YouTube-style player
    st.video(temp_video_path)

    if st.button("Analyze Video", type="primary"):
        model = load_model()
        if model:
            st.markdown("---")
            st.subheader("Analysis Results")
            
            progress_bar = st.progress(0, text="Starting analysis...")
            status_text = st.empty()
            results_placeholder = st.empty()

            for update in predict_video(temp_video_path, model):
                if update["status"] == "Processing":
                    progress_text = f"Processing sequence {update['sequence']} of {update['total']}..."
                    progress_bar.progress(update["progress"], text=progress_text)
                
                elif update["status"] == "Complete":
                    progress_bar.progress(1.0, text="Analysis Complete!")
                    time.sleep(1) # Give a moment for the user to see "complete"
                    progress_bar.empty()

                    final_pred = update["final_prediction"]
                    prediction_text = "FAKE" if final_pred == 1 else "REAL"
                    color = "red" if final_pred == 1 else "green"

                    with results_placeholder.container():
                        st.markdown(f"### Final Prediction: <span style='color:{color};'>{prediction_text}</span>", unsafe_allow_html=True)
                        st.write("This prediction is based on a majority vote from the analyzed video sequences.")
                        st.write(f"**Sequence-level Predictions (0=Real, 1=Fake):** `{update['sequence_predictions']}`")
                        
    # Clean up the temporary file
    if 'temp_video_path' in locals() and os.path.exists(temp_video_path):
        os.remove(temp_video_path)

else:
    st.info("Please upload a video file using the sidebar on the left.")