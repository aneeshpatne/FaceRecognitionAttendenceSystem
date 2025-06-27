# 🎯 Face Recognition Attendance System

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)](https://scikit-learn.org/)

A sophisticated real-time attendance management system powered by advanced facial recognition technology. This system leverages multiple machine learning algorithms to provide accurate face detection and recognition with comprehensive attendance tracking capabilities.

## ✨ Key Features

### 🔍 Advanced Face Recognition

- **Real-time face detection** using OpenCV and webcam integration
- **High-accuracy face recognition** with face_recognition library
- **Multi-algorithm support** with performance comparison
- **Unknown face detection** with confidence thresholds

### 🤖 Machine Learning Models

- **K-Nearest Neighbors (KNN)** - Primary classifier with distance weighting
- **Support Vector Machine (SVM)** - Linear kernel with probability estimation
- **Random Forest** - Ensemble method for robust predictions
- **Naive Bayes** - Gaussian implementation for baseline comparison

### 📊 Attendance Management

- **Entry/Exit tracking** with timestamp logging
- **CSV-based records** for easy data analysis
- **Date and time stamping** with automatic formatting
- **User confirmation system** to prevent false entries
- **Duplicate prevention** and validation checks

### 🎛️ User Interface

- **Interactive webcam display** with real-time face detection boxes
- **Visual feedback** with name labels and confidence indicators
- **Console-based controls** for entry/exit selection
- **Error handling** and user-friendly prompts

## 🏗️ Project Architecture

```
FaceRecognitionAttendanceSystem/
├── 📁 classifier/                  # Trained model storage
│   └── trained_knn_model.clf      # Serialized KNN model
├── 📁 train_image/                 # Training dataset
│   ├── person1/                    # Individual person folders
│   ├── person2/                    # Multiple images per person
│   └── ...                         # Organized by name
├── 📄 FaceRecogKnn.py             # Main recognition engine
├── 📄 Train_main.py               # Model training pipeline
├── 📄 attendance.csv              # Attendance records
└── 📄 README.md                   # Documentation
```

## 🚀 Quick Start

### Prerequisites

Ensure you have Python 3.7+ installed, then install the required dependencies:

```bash
pip install opencv-python
pip install face-recognition
pip install scikit-learn
pip install pandas
pip install numpy
pip install matplotlib
```

### Installation & Setup

1. **Clone or download** this repository to your local machine

2. **Prepare training data:**

   ```bash
   # Create individual folders for each person
   mkdir train_image/john_doe
   mkdir train_image/jane_smith

   # Add 5-10 clear face images per person
   # Images should be well-lit with clear facial features
   ```

3. **Train the models:**

   ```bash
   python Train_main.py
   ```

   This will:

   - Process all training images
   - Extract facial encodings
   - Train multiple ML models
   - Display accuracy metrics
   - Save the best performing model

4. **Run the attendance system:**
   ```bash
   python FaceRecogKnn.py
   ```

## 🔧 Detailed Usage

### Training New Users

1. Create a folder named with the person's full name (e.g., `john_doe`)
2. Add 5-10 high-quality face images:
   - Well-lit environment
   - Clear facial features
   - Different angles and expressions
   - Consistent lighting conditions
3. Run the training script to update models

### Recording Attendance

1. Launch the recognition system
2. Position yourself in front of the webcam
3. Wait for face detection and recognition
4. Select option:
   - `1` for Entry time
   - `2` for Exit time
5. Confirm the detected identity
6. Attendance is automatically logged

### Monitoring Performance

The system provides real-time feedback:

- **Green box**: Successfully recognized face
- **Red text**: Person's name with confidence
- **Console output**: Recognition status and options

## 📈 Model Performance

The system trains and evaluates four different algorithms:

| Algorithm         | Typical Accuracy | Strengths                                         |
| ----------------- | ---------------- | ------------------------------------------------- |
| **KNN**           | 85-95%           | Fast prediction, good with small datasets         |
| **SVM**           | 80-90%           | Robust to outliers, works well in high dimensions |
| **Random Forest** | 75-85%           | Handles overfitting, provides feature importance  |
| **Naive Bayes**   | 70-80%           | Fast training, good baseline performance          |

### Performance Optimization Tips

- **Image Quality**: Use high-resolution, well-lit images
- **Dataset Size**: Minimum 5 images per person, ideally 10+
- **Consistency**: Maintain similar lighting and camera conditions
- **Threshold Tuning**: Adjust recognition threshold based on accuracy needs

## 📊 Attendance Data Format

The system generates structured attendance records:

```csv
Name,Date,Time,Entry/Exit
john_doe,27/6/2025,09:15:32,Entry
jane_smith,27/6/2025,09:16:45,Entry
john_doe,27/6/2025,17:30:15,Exit
jane_smith,27/6/2025,17:32:08,Exit
```

### Data Fields

- **Name**: Recognized person identifier
- **Date**: DD/MM/YYYY format
- **Time**: HH:MM:SS format (24-hour)
- **Entry/Exit**: Attendance type

## 🛠️ Technical Implementation

### Core Components

#### Face Detection Pipeline

```python
# Frame processing workflow
frame → resize → color_convert → face_locations → face_encodings → prediction
```

#### Recognition Algorithm

- Uses dlib's face recognition model
- Generates 128-dimensional face encodings
- Applies distance-based matching with KNN
- Implements confidence thresholds for unknown faces

#### Training Process

- Automatically processes training images
- Extracts facial features using face_recognition
- Splits data into train/test sets (80/20)
- Trains multiple classifiers simultaneously
- Evaluates and compares model performance

## 🔧 Configuration Options

### Recognition Threshold

Adjust in `FaceRecogKnn.py`:

```python
threshold = 0.6  # Lower = more strict, Higher = more lenient
```

### Model Parameters

Customize in `Train_main.py`:

```python
n_neighbors = 2        # KNN neighbors
test_size = 0.2       # Train/test split ratio
max_depth = 2         # Random Forest depth
```

## 🐛 Troubleshooting

### Common Issues

**No face detected:**

- Ensure good lighting
- Position face clearly in camera view
- Check camera permissions

**Low accuracy:**

- Add more training images
- Improve image quality
- Retrain the model

**Recognition errors:**

- Adjust threshold values
- Add more diverse training data
- Check for similar-looking individuals

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your improvements
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open-source and available under the MIT License.

## 🙏 Acknowledgments

- **face_recognition** library by Adam Geitgey
- **OpenCV** for computer vision capabilities
- **scikit-learn** for machine learning algorithms
- **dlib** for facial landmark detection

---

_Built with ❤️ for efficient attendance management_
