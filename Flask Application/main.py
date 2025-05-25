import os
from flask import Flask, request, render_template
import torch
from model import Model
from utils import extract_frames, validation_dataset, make_prediction
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import face_recognition

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(2).to(device)
state_dict = torch.load("model_ram.pt", map_location=torch.device('cpu'))
model.load_state_dict(state_dict)  # load weights into model
model.eval()  # set model to eval mode

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["video"]
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    video_path = os.path.normpath(file_path)
    file.save(file_path)

    # Code for making prediction
    im_size = 112
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])
    video_dataset = validation_dataset([video_path], sequence_length=20, transform=train_transforms)
    print("file_path is", video_path)
    #print("ram...",video_dataset[0].shape)
    try:
        video_tensor = video_dataset[0]
    except RuntimeError as e:
        return render_template("index.html", result="Failed to process video: " + str(e), video_path="/" + file_path)
    prediction = make_prediction(model, video_tensor, './')
    if prediction[0] == 1:
        label = "REAL"
    else:
        label = "FAKE"
    return render_template("index.html", result=label, video_path="/" + file_path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)