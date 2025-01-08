# Face Recognition Project

## Overview
This project focuses on facial recognition using feature extraction and classification techniques. The workflow includes creating a custom dataset, cropping face regions using DeepFace, augmenting the cropped images, extracting features using the MLP model, training the model, and performing real-time face recognition with a webcam.

## Project Workflow

### 1. Data Preprocessing
- **Dataset Creation**: The dataset is created by collecting face images and organizing them into folders where each folder represents a person.
- **Face Cropping**: Faces are cropped from images using the `DeepFace` library before further processing.
- **Data Augmentation**: After cropping the faces, augmentation techniques (such as rotation, flipping, and scaling) are applied to increase dataset diversity.
- **Label Encoding**: Folder names representing individuals are encoded into numerical labels using `LabelEncoder`.
- **Feature Extraction**: Features are extracted from the augmented, cropped images using the `DeepFace` library with the MLP model. These features are then saved into a CSV file for training and testing.

### 2. Model Training
- **Algorithm**: Multilayer Perceptron (MLP) is used for classification.
- **Training Accuracy**: The model achieves an accuracy of 91.64% on the validation set.
- **Model Storage**: The trained model is saved in `.h5` format for future use.

### 3. Real-Time Face Recognition
- Real-time face recognition is implemented using a webcam.
- When `r` buttom pressed the model run a face detection, the corresponding person's name is displayed on the log. 

## Setup and Installation

### Environment Setup
1. **Install Anaconda**:
   - Download and install Anaconda from [anaconda.com](https://www.anaconda.com/products/distribution).

2. **Create a Conda Environment**:
   ```bash
   conda create --name face_recognition python=3.8
   conda activate face_recognition
   ```

3. **Install Required Libraries**:
   ```bash
   pip install -r requirementsDeepface.txt
   ```

### Dataset Setup
- Place face images in a directory structure where folder names represent the person’s name. Example:
  ```
  Data/
  |-- Person_A/
  |   |-- img1.jpg
  |   |-- img2.jpg
  |-- Person_B/
      |-- img1.jpg
      |-- img2.jpg
  ```

## Usage

### Training the Model
1. Run the feature extraction script to save the features into a CSV file.
2. Train the MLP model using the preprocessed and augmented features.
3. Save the trained model to an `.h5` file.

### Real-Time Recognition
1. Load the trained model and label encoder.
2. Run the real-time face recognition script.
3. The script will detect and recognize faces through the webcam and display the corresponding names.

## Example Output

### Training Results:
- **Loss**: `0.4977`
- **Accuracy**: `85.21%`

### Real-Time Recognition Output:
When a face is recognized, the output will be displayed as follows:
```text
Detected: Person_A
```

## Challenges and Reflections

### Challenges:
- **Preprocessing Consistency**: Ensuring consistent preprocessing for both training and real-time inference.
- **Face Detection**: Managing mislabeled or undetected faces during feature extraction.

### Reflections:
- The combination of `DeepFace` for face cropping and feature extraction, along with `MLP` for classification, provides robust recognition results.
- Real-time performance depends on the quality of input images and hardware capabilities.

## Future Work
- Enhance model accuracy by augmenting the dataset.
- Incorporate additional pre-trained models for feature extraction.
- Optimize the system for faster real-time inference.

---

Feel free to contribute or provide feedback to improve this project!

---
