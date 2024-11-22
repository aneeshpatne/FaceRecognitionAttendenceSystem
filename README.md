# Face Recognition Attendance System

A real-time attendance system using facial recognition with KNN, SVM, Random Forest, and Naive Bayes classifiers.

## Features

- Face detection and recognition using webcam
- Multiple classification models:
    - K-Nearest Neighbors (KNN)
    - Support Vector Machine (SVM)
    - Random Forest
    - Naive Bayes
- Attendance logging with entry/exit tracking
- CSV-based attendance records
- Training interface for new faces

## Project Structure

```
.
├── FaceRecogKnn.py      # Main recognition script
├── Train_main.py        # Model training script
├── attendance.csv       # Attendance log file
├── classifier/          # Trained model storage
└── train_image/         # Training images directory
```

## Requirements

- Python 3.x
- OpenCV (cv2)
- face_recognition
- scikit-learn
- pandas
- numpy
- matplotlib

## Usage

1. Add training images:
     - Create a folder with the person's name under `train_image`
     - Add multiple face images of the person in their folder

2. Train the model:
     ```python
     python Train_main.py
     ```

3. Run attendance system:
     ```python
     python FaceRecogKnn.py
     ```

## How It Works

- `train()` function trains multiple classifiers using face encodings
- `predict()` detects faces and predicts identity
- `record_attendance()` logs attendance with timestamp
- System supports both entry and exit time tracking
- Real-time face detection with OpenCV webcam integration

## Attendance Format

The `attendance.csv` file tracks:
- Name
- Date
- Time
- Entry/Exit status

## Performance

The system evaluates accuracy across multiple classifiers:
- KNN
- SVM
- Random Forest
- Naive Bayes

Model performance metrics are displayed during training.


