

# 💓 Heart Failure Prediction

### A machine learning web application that predicts the likelihood of heart failure based on patient health data.
### Goal: Support healthcare professionals in early detection to improve patient outcomes.

# 📊 Overview
1. Trained on real-world clinical data from Kaggle Heart Failure Dataset.

2. Performed data preprocessing, feature engineering, and model optimization to achieve high recall for early detection.

3. Deployed with Streamlit for interactive, real-time predictions.

# 🔍 Key Features
1. Data Cleaning & Feature Engineering – handled missing values, outlier treatment, and categorical encoding.

2. Model Training & Evaluation – compared Logistic Regression, Random Forest, and XGBoost; selected the best model based on recall.

3. Interactive Web Interface – user-friendly form with dropdowns and numeric inputs for predictions.

4. Deployment Ready – runs locally or can be hosted on Streamlit Cloud.


# 🛠️ Tech Stack
1. Python 3.10+

2. Streamlit (Web Framework)

3. Pandas, NumPy, Scikit-learn (Data Pre-processing & Machine Learning Algorithm)

# 📈 Results
1. Best Model: Random Forest Classifier

2. Performance: Recall – 0.92, Accuracy – 0.88

3. Why Recall Matters: In healthcare, false negatives can cost lives, so the model focuses on correctly identifying positive cases.


# 📋 Prerequisites
### Make sure you have the following installed:

1. [Github Account](https://github.com)
2. [VS Code IDE](https://code.visualstudio.com/)
3. [Git CLI](https://git-scm.com/downloads)


# ⚙️ Step for program Installation

## 1️⃣ Clone the repository

#### open command prompt from your browser then type in
#### git clone https://github.com/Phuoctram2412/heartfailureprediction.git 
#### cd heartfailureprediction


## 2️⃣ Install dependencies

#### pip install -r requirements.txt

## 3️⃣ Run the App

#### streamlit run app.py


# 👤 Author’s Role
1. Developed the project end-to-end:

2. Collected and preprocessed data

3. Engineered features

4. Trained, tuned, and evaluated models (focus more on Recall)

5. Deployed a functional web application