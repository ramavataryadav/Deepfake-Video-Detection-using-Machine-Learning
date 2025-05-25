import os
from flask import Flask, request, render_template
import torch
from model import DeepFakeModel
from utils import extract_frames

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepFakeModel().to(device)
model.load_state_dict(torch.load("model_ram.pt", map_location=device))
model.eval()

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["video"]
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    try:
        frames = extract_frames(file_path).to(device)
        with torch.no_grad():
            output = model(frames)
            pred = torch.argmax(output, dim=1).item()
            label = "REAL" if pred == 1 else "FAKE"
    except Exception as e:
        label = f"Error: {e}"

    return render_template("index.html", result=label, video_path="/" + file_path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
