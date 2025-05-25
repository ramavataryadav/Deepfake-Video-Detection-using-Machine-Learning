# Deepfake-Video-Detection-using-Machine-Learning
# 1. Introduction
In today’s world of growing social media, deepfake videos are a big threat as they can be used for fake news, blackmail, or political harm. These videos are made using apps like FaceApp and Face Swap that use AI. Our system helps detect deepfakes by using a CNN model (ResNeXt) to get important features from video frames. Then, an LSTM model checks the video frame-by-frame to decide if it’s real or fake. We trained the system using well-known deepfake datasets like FaceForensics++, Celeb-DF, and DFD. A simple web app lets users upload videos and get results showing if the video is deepfake or real, along with a confidence score.

# 2. Architecture: Deepfake Detection Pipeline
1. Input Interface
   Video Input Module
     . Upload Interface (User-uploaded video)
     . Dataset Loader (Labeled Fake/Real training videos)

3. Preprocessing Module
   Frame Extractor
     . Converts videos to frame sequences
   Face Analyzer
     . Face Detection
     . Face Alignment & Cropping
Preprocessed Face Frame Storage
  . Stores face-only frames for downstream processing

5. Dataset Handling
Data Organizer
  . Organizes data into training, validation, and test sets
Batch Generator
  . Loads face sequences with labels in batches for training and evaluation

6. Feature Engineering
Feature Extraction Module
  . ResNext / EfficientNet
  . Converts face frames to deep feature representations

7. Temporal Modeling
Sequence Modeling
  . LSTM / Transformer
  . Captures temporal dependencies across frames

8. Classification
Video Classification Head
  . Final prediction: Real or Fake

9. Training & Evaluation
Training Controller
  . Orchestrates model training and validation
Metrics Evaluator
  . Generates confusion matrix, accuracy, precision, recall

10. Deployment & Inference
Model Exporter
  . Saves the trained model
Inference Engine
  . Loads the model for prediction
  . Takes uploaded video, preprocesses, predicts: REAL / FAKE

Flows
Training Flow – Dataset → Preprocessing → Feature + Temporal Modeling → Training + Evaluation

Prediction Flow – Upload Video → Preprocessing → Feature + Temporal Modeling → Predit
