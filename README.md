# 🧠 Deep Learning for Video Classification 🚀🎥

This project focuses on video and image classification using advanced **Deep Learning** models. By leveraging architectures such as **3D CNNs**, **VGG16**, and **EfficientNet**, it aims to classify input data accurately while optimizing computational performance. The project demonstrates effective use of feature extraction, regularization, and fine-tuning techniques.

---

## 📋 Features

- **Preprocessing and Augmentation**:
  - Prepares video and image datasets for model training.
  - Applies augmentation techniques for robust training.

- **Model Architectures**:
  - **3D CNNs** for temporal and spatial feature extraction.
  - **VGG16** and **EfficientNetB2** for feature transfer and fine-tuning.
  - Incorporates **Batch Normalization**, **Dropout**, and **Global Average Pooling** for regularization and dimensionality reduction.

- **Evaluation and Visualization**:
  - Generates confusion matrices and classification reports.
  - Visualizes training performance and feature importance.

---

## 🛠️ Model Architecture 💡

### 1. **Convolutional Blocks**
- **4 Convolutional Blocks** with increasing filter sizes (32, 64, 128, 256).
- Each block includes:
  - 3D Convolutional layers with **swish activation**.
  - **Batch Normalization** for stable training.
  - **SpatialDropout3D** for regularization.
  - **MaxPooling3D** for dimensionality reduction.

### 2. **Fine-Tuning VGG16**
- **Pretrained VGG16**:
  - Freezes trainable layers.
- Additional Layers:
  - **Global Average Pooling** to condense feature maps.
  - Fully connected **Dense Layers** with 1024, 512, and 256 neurons.
  - **Batch Normalization** and **Dropout** (rates: 0.6, 0.5, 0.4).
  - **Output Layer**:
    - Single neuron with **sigmoid activation** for binary classification.

---

## 📂 Project Structure

- **`DeepLearningProject-FinalVersion.ipynb`**: Jupyter Notebook with the full implementation.
- **`data/`**: Directory containing video/image datasets.

---

## 🛠️ Technologies Used

- **TensorFlow/Keras**: For model design, training, and evaluation.
- **Mediapipe**: For preprocessing and extracting features.
- **Pandas** and **NumPy**: For data manipulation and preprocessing.
- **Matplotlib** and **Seaborn**: For visualization.
- **OpenCV**: For handling video and image data.

---

## 🌟 Contributors

- **Sana Araj**
- **Shahad Adel**
- **Sara Thaer**
- **Deem Alrashidi**
- **Sahar Alshehri**

