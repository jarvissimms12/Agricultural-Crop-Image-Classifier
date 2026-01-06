# 1. Clone your repository (Replace with your actual GitHub URL)
git clone https://github.com/jarvissimms12/Image-Recognition-Model.git
cd Your-Repo-Name

# 2. Install the necessary Python libraries
pip install tensorflow keras split-folders opencv-python numpy matplotlib

# 3. Download the dataset directly from Kaggle
# Dataset Link: https://www.kaggle.com/datasets/mdwaquarazam/agricultural-crops-image-classification
kaggle datasets download -d mdwaquarazam/agricultural-crops-image-classification

# 4. Unzip the downloaded dataset
unzip agricultural-crops-image-classification.zip -d Agricultural-crops

# 5. Run your training script
python main.py
