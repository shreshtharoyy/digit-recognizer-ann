# ✍️ Handwritten Digit Recognition (ANN)

A web-based handwritten digit recognition system built using an **Artificial Neural Network (ANN)** trained on the **MNIST dataset** and deployed with **Streamlit** for real-time predictions.

---

## 🚀 Features

* Interactive canvas for digit input (0–9)
* Real-time prediction using trained ANN model
* ~98% validation accuracy on MNIST
* Input preprocessing aligned with training data

---

## 📂 Project Structure

```id="q6j3u1"
├── app.py               # Streamlit application (UI + inference)
├── train_model.py       # Model training script
├── model.keras          # Trained ANN model
├── requirements.txt     # Dependencies
└── README.md
```

---

## 📊 Dataset

The model is trained on the **MNIST dataset (CSV format)**:

🔗 https://www.kaggle.com/datasets/oddrationale/mnist-in-csv

### Details:

* Total samples: 70,000
* Training: 60,000
* Testing: 10,000
* Image size: 28 × 28 (grayscale)
* Flattened input: 784 features

Each row consists of:

* Label (0–9)
* 784 pixel values (0–255)

### Rationale for Using CSV Format

The CSV format was used to:

* Simplify data loading using Pandas
* Enable efficient training of fully connected (ANN) models
* Avoid additional image-loading overhead during experimentation

---

## 🧠 Model Architecture

The neural network was **designed and trained using TensorFlow/Keras**:

```id="3k1l4y"
Input (784)
  → Dense(256, ReLU)
  → Dense(128, ReLU)
  → Dense(64,  ReLU)
  → Dense(10,  Softmax)
```

### Training Configuration

* Epochs: 20
* Batch size: 32
* Validation accuracy: ~98%

---

## ⚙️ Preprocessing Pipeline

To ensure consistency between user input and training data, the application includes an input preprocessing pipeline:

```id="lbb0l8"
RGBA canvas (280×280)
  → Grayscale conversion
  → Binary thresholding
  → Bounding box cropping
  → Resize to 20×20
  → Center padding to 28×28
  → Gaussian blur
  → Normalization (pixel / 255.0)
```

This pipeline was **integrated and adapted** to align real-time input with the MNIST data distribution.

---

## ▶️ Running Locally

### Install dependencies

```bash id="bq7mxr"
pip install -r requirements.txt
```

### Run the application

```bash id="9fw9ps"
streamlit run app.py
```

---

## 🧰 Tech Stack

* TensorFlow / Keras
* Pandas
* OpenCV
* Streamlit

---

## ⚠️ Common Issues

| Issue                                                  | Cause                                           | Resolution                                                                  |
| ------------------------------------------------------ | ----------------------------------------------- | --------------------------------------------------------------------------- |
| Incorrect predictions despite high validation accuracy | Mismatch between training data and input format | Align preprocessing pipeline with MNIST (centering, scaling, normalization) |
| Low confidence predictions                             | Input distribution mismatch during inference    | Ensure normalization, centering, and scaling match MNIST format             |
| Model performs well in training but poorly in app      | Inconsistent preprocessing during deployment    | Replicate training data conditions in inference pipeline                    |

---

## 📌 Notes

* Model architecture, training, and evaluation were performed independently
* Deployment layer was implemented using existing tools and adapted for integration
* Focus was placed on ensuring consistency between training and inference pipelines

---

## 🚀 Future Work

* Replace ANN with CNN for improved performance

