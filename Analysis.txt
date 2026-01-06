# Image-Recognition-Model

This project uses a Convolutional Neural Network (CNN) to process visual data and provide instant classifications. I built this to explore how machines "see" patterns in pixels and to create a tool that could group images based on similar features. Results Accuracy: Achieved 82% on the test set.

🌾 Agricultural Crop Image Classifier!

🚀 Project Overview

This project is a deep learning application designed to identify 30 different types of agricultural crops using computer vision. By leveraging Transfer Learning with the ResNet50 architecture, the model can classify crop images with high accuracy, providing a foundation for automated crop monitoring and precision agriculture.

🔍 Usage & Inference
The model includes a prediction pipeline that:
1.  **Standardizes** input images to 224x224.
2.  **Preprocesses** pixels using the ResNet50 distribution requirements.
3.  **Outputs** the predicted crop category along with a confidence percentage.

To test a single image:
Python
prediction = predict_img('path_to_leaf.jpg', model)
print(prediction) 
# Output: "Coffee-plant (98.45%)"

🧠 Model Architecture

The system uses a state-of-the-art Convolutional Neural Network (CNN) pipeline:
Base Model: ResNet50 (pre-trained on ImageNet) for robust feature extraction.
Global Average Pooling: Used to reduce the spatial dimensions of the feature maps.
Dense Layer: 128 neurons with ReLU activation for learning specific agricultural patterns.
Regularization: Dropout layer (0.5) to prevent overfitting and improve generalization.
Output: 30-way Softmax layer for categorical classification.

📊 Performance Results

The model was trained over 100-300 epochs, demonstrating consistent learning and error reduction.
Final Test Accuracy: 81.90%
Training Accuracy: ~96%
Validation Accuracy: ~88%

Training Visualization

🛠️ Tech Stack

Language: Python
Frameworks: TensorFlow, Keras
Tools: OpenCV, Matplotlib, Split-Folders
Environment: Kaggle / Google Colab

📂 Dataset Information
The model classifies 30 crop categories, including: Cherry, Coffee-plant, Cucumber, Lemon, Olive-tree, Rice, Soybean, Sugarcane, Wheat, and more.
