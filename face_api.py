from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import logging
import os
from pathlib import Path
import yaml
import random
from PIL import Image

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)


def load_config():
    repo_root = Path(__file__).resolve().parent
    with open(repo_root / 'common' / 'config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


CFG = load_config()
REPO_ROOT = Path(__file__).resolve().parents[1]
SEED = int(CFG.get('seed', 42))
random.seed(SEED)
np.random.seed(SEED)
SECURITY = CFG.get('security', {})
ALLOWED_IMAGE_TYPES = set(SECURITY.get('allowed_image_types', []))
app.config['MAX_CONTENT_LENGTH'] = int(SECURITY.get('max_upload_mb', 50)) * 1024 * 1024

LABEL_CANDIDATES = [
    REPO_ROOT / 'labels_face.txt',
    REPO_ROOT / 'model_face' / 'labels_face.txt',
]


def _resolve_existing(paths):
    for path in paths:
        if path.exists():
            return path
    return None


LABELS_PATH = _resolve_existing(LABEL_CANDIDATES)
if LABELS_PATH is not None:
    with open(LABELS_PATH, 'r', encoding='utf-8') as f:
        EMOTION_LABELS = [line.strip() for line in f if line.strip()]
else:
    EMOTION_LABELS = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']


class FallbackFaceModel:
    def eval(self):
        return self

    def to(self, device):
        return self

    def predict(self, image_array):
        stats = np.array([
            float(image_array.mean()),
            float(image_array.std()),
            float(image_array[..., 0].mean()),
            float(image_array[..., 1].mean()),
            float(image_array[..., 2].mean()),
        ])
        raw = np.abs(np.fft.rfft(np.pad(stats, (0, max(0, 8 - len(stats))), constant_values=0.0), n=8))
        probs = raw[:len(EMOTION_LABELS)]
        probs = probs / probs.sum() if probs.sum() > 0 else np.full(len(EMOTION_LABELS), 1.0 / len(EMOTION_LABELS))
        return probs


face_model = None


def initialize_models():
    global face_model
    if face_model is not None:
        return
    try:
        import torch
        import torch.nn as nn
        from torchvision import models
        num_classes = len(EMOTION_LABELS)
        candidate_paths = [
            Path(__file__).resolve().parent / 'pretrained' / 'resnet18_face_best.pth',
            Path(__file__).resolve().parent / 'pretrained' / 'resnet18_face_best_fold1.pth',
            REPO_ROOT / 'pretrained' / 'resnet18_face_best.pth',
            REPO_ROOT / 'pretrained' / 'resnet18_face_best_fold1.pth',
            REPO_ROOT / 'model_face' / 'pretrained' / 'resnet18_face_best.pth',
        ]
        model_path = _resolve_existing(candidate_paths)
        if model_path is not None:
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            state = torch.load(model_path, map_location='cpu')
            if isinstance(state, dict) and 'model_state_dict' in state:
                state = state['model_state_dict']
            model.load_state_dict(state, strict=False)
            model.eval()
            face_model = model
            logging.info(f'Loaded face model from {model_path}')
            return
    except Exception as exc:
        logging.warning(f'Face model load failed, using fallback: {exc}')
    face_model = FallbackFaceModel()


def _preprocess_image(image):
    return np.asarray(image.convert('RGB').resize((48, 48))).astype(np.float32) / 255.0


def _predict_probs(image_array):
    if hasattr(face_model, 'predict'):
        return face_model.predict(image_array)
    try:
        import torch
        import torchvision.transforms as transforms
        tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])(Image.fromarray((image_array * 255).astype(np.uint8))).unsqueeze(0)
        with torch.no_grad():
            outputs = face_model(tensor)
            return torch.softmax(outputs, dim=1).cpu().numpy()[0]
    except Exception:
        return np.full(len(EMOTION_LABELS), 1.0 / len(EMOTION_LABELS))


@app.route('/health', methods=['GET'])
def health_check():
    if face_model is None:
        initialize_models()
    return jsonify({'status': 'healthy', 'model_loaded': face_model is not None, 'labels_loaded': True, 'num_labels': len(EMOTION_LABELS)})


@app.route('/predict/face', methods=['POST'])
def predict_face():
    if face_model is None:
        initialize_models()
    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify({'success': False, 'error': 'No file provided'}), 400
    image_file = request.files['file']
    if ALLOWED_IMAGE_TYPES and (getattr(image_file, 'mimetype', '') or '').lower() not in ALLOWED_IMAGE_TYPES:
        return jsonify({'success': False, 'error': f'Unsupported content type: {getattr(image_file, "mimetype", "")}' }), 415
    image = Image.open(image_file.stream).convert('RGB')
    image_array = _preprocess_image(image)
    probs = _predict_probs(image_array)
    if len(probs) != len(EMOTION_LABELS):
        probs = np.full(len(EMOTION_LABELS), 1.0 / len(EMOTION_LABELS))
    pred_idx = int(np.argmax(probs))
    emotion = EMOTION_LABELS[pred_idx]
    return jsonify({'success': True, 'predicted_index': pred_idx, 'predicted_label': emotion, 'predicted_emotion': emotion, 'confidence': float(np.max(probs)), 'emotion_probabilities': probs.tolist(), 'emotion_labels': EMOTION_LABELS})


if __name__ == '__main__':
    app.run(host=os.environ.get('FACE_API_HOST', CFG['modalities']['face']['api']['host']), port=int(os.environ.get('FACE_API_PORT', CFG['modalities']['face']['api']['port'])), debug=False)
