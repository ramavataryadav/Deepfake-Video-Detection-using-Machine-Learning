# Deepfake-Video-Detection-using-Machine-Learning

## 1. Introduction

In today’s digital age, deepfake videos pose a significant threat due to their potential misuse in spreading misinformation, blackmail, and political manipulation. Deepfakes are synthetic media created using advanced AI techniques and applications such as FaceApp and Face Swap.

This project aims to detect deepfake videos using a two-stage machine learning approach:
- **ResNeXt (CNN)**: Extracts robust spatial features from individual video frames.
- **LSTM (RNN)**: Captures temporal dependencies across frames to classify the video as real or fake.

We trained our models using publicly available and widely accepted datasets like:
- **FaceForensics++**
- **Celeb-DF**
- **DFD (Deepfake Detection Dataset)**

A simple **web application** allows users to upload videos and receive real-time predictions, including a **confidence score** indicating the likelihood of the video being fake or real.

---

## 2. Architecture: Deepfake Detection Pipeline

### 1. **Input Interface**
- **Video Input Module**:
  - Upload Interface (User-uploaded video)
  - Dataset Loader (Labeled Fake/Real training videos)

### 2. **Preprocessing Module**
- **Frame Extractor**:
  - Converts videos into individual frame sequences
- **Face Analyzer**:
  - Detects faces in each frame
  - Performs face alignment and cropping
- **Preprocessed Face Frame Storage**:
  - Saves face-only frames for further processing

### 3. **Dataset Handling**
- **Data Organizer**:
  - Splits data into training, validation, and test sets
- **Batch Generator**:
  - Loads batches of face sequences with labels for model training and evaluation

### 4. **Feature Engineering**
- **Feature Extraction Module**:
  - Uses **ResNeXt** or **EfficientNet** to convert face frames into high-level feature representations

### 5. **Temporal Modeling**
- **Sequence Modeling**:
  - Utilizes **LSTM** or **Transformer** networks to analyze the sequence of features over time

### 6. **Classification**
- **Video Classification Head**:
  - Outputs final classification: **Real** or **Fake**

### 7. **Training & Evaluation**
- **Training Controller**:
  - Manages training loop and optimization
- **Metrics Evaluator**:
  - Computes performance metrics (confusion matrix, accuracy, precision, recall)

### 8. **Deployment & Inference**
- **Model Exporter**:
  - Saves the final trained model
- **Inference Engine**:
  - Loads the trained model
  - Takes uploaded videos, performs preprocessing, and predicts: **REAL / FAKE**

---

## 3. Flow Overview

### 🔁 **Training Flow**

