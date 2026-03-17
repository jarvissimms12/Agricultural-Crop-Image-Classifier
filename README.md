🌾 Agricultural Crop Image Classifier
![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-81.9%25-brightgreen)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
A deep learning image classification system that identifies 30 distinct agricultural crop categories from photographs using Transfer Learning with ResNet50. Achieved 81.90% test accuracy on a challenging multi-class dataset.
---
🎯 Project Highlights
Metric	Result
Test Accuracy	81.90%
Architecture	ResNet50 (Transfer Learning)
Classes	30 agricultural crop categories
Technique	Transfer Learning + Fine-tuning
Regularization	Dropout to minimize overfitting
Data Pipeline	ImageDataGenerator with real-time augmentation
---
🧠 Technical Approach
Why ResNet50?
ResNet50's residual connections solve the vanishing gradient problem in deep networks, making it ideal for extracting rich visual features from complex crop imagery — without training from scratch.
Architecture Design
```
Input (224×224×3)
    → ResNet50 Base (ImageNet weights, frozen)
    → Global Average Pooling
    → Dense(256, ReLU)
    → Dropout(0.5)          ← prevents overfitting
    → Dense(30, Softmax)    ← 30 crop classes
```
Data Pipeline
ImageDataGenerator — real-time augmentation (rotation, flip, zoom, shear)
Train/Val/Test split — stratified to maintain class balance
Normalization — pixel values rescaled to [0, 1]
---
📊 Results
```
Final Test Accuracy:  81.90%
Final Test Loss:      0.72

Training Accuracy:    ~94%   (after fine-tuning)
Validation Accuracy:  ~83%
```
The gap between training and validation accuracy indicates moderate overfitting, controlled via Dropout regularization and data augmentation. Further improvement could be achieved with additional augmentation or a larger dataset.
---
🚀 How to Run
```bash
# 1. Clone the repo
git clone https://github.com/jarvissimms12/Agricultural-Crop-Image-Classifier.git
cd Agricultural-Crop-Image-Classifier

# 2. Install dependencies
pip install tensorflow keras split-folders opencv-python numpy matplotlib

# 3. Download the dataset from Kaggle
# Dataset: https://www.kaggle.com/datasets/mdwaquarazam/agricultural-crops-image-classification
kaggle datasets download -d mdwaquarazam/agricultural-crops-image-classification
unzip agricultural-crops-image-classification.zip -d Agricultural-crops

# 4. Train the model
python main.py
```
---
📁 Project Structure
```
Agricultural-Crop-Image-Classifier/
├── main.py             # Full pipeline: data prep → model build → train → evaluate
├── requirements.txt    # Python dependencies
├── Analysis.txt        # Detailed run logs and evaluation notes
└── README.md
```
---
🔑 Key Skills Demonstrated
Deep Learning — CNN architecture design and training
Transfer Learning — ResNet50 with custom classification head
Data Engineering — ImageDataGenerator pipeline with augmentation
Model Evaluation — accuracy/loss curves, overfitting analysis
Python — TensorFlow, Keras, NumPy, matplotlib
Reproducibility — modular `main.py`, documented setup instructions
---
🔗 Related Projects
Media Content Analytics Dashboard — Interactive BI dashboard with Streamlit & Plotly
NYC Housing Data Analysis — Real-world data pipeline with NYC Open Data API
Credit Card Fraud Detection — Imbalanced classification with SMOTE + Random Forest
---
👤 Author
Jarvis Simms | MS Data Science, NYIT 2025 | Brooklyn, NY  
GitHub • LinkedIn
