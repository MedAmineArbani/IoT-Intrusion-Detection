# IoT Intrusion Detection System

A comprehensive machine learning project for detecting intrusions in IoT networks using the RT-IoT2022 dataset. This project implements multiple classification algorithms including XGBoost, Random Forest, Decision Tree, K-Nearest Neighbors, Support Vector Machine, Naive Bayes, Logistic Regression, and TabNet to identify various types of network attacks.

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Models Implemented](#models-implemented)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project focuses on building and evaluating machine learning models to detect intrusions in IoT networks. The system classifies network traffic into different attack types including DDoS, MITM, Scanning, and Normal traffic.

### Key Features
- Multiple ML algorithms for comparison
- Comprehensive data preprocessing pipeline
- Feature engineering and scaling
- Model evaluation with multiple metrics
- Visualization of results and feature importance
- Saved models for deployment

## 📊 Dataset

The project uses the **RT-IoT2022** dataset, which contains network traffic data from IoT devices with various attack scenarios.

### Attack Types
- **DDoS** (Distributed Denial of Service)
- **MITM** (Man-in-the-Middle)
- **Scanning**
- **Normal** traffic

### Dataset Features
- Multiple network traffic features
- Labeled attack types
- Balanced/Imbalanced class distribution
- Real-world IoT network scenarios

## 🤖 Models Implemented

1. **XGBoost** - Gradient boosting ensemble method
2. **Random Forest** - Ensemble of decision trees
3. **Decision Tree** - Single tree-based classifier
4. **K-Nearest Neighbors (KNN)** - Instance-based learning
5. **Support Vector Machine (SVM)** - Kernel-based classifier
6. **Naive Bayes** - Probabilistic classifier
7. **Logistic Regression** - Linear classification model
8. **TabNet** - Attention-based deep learning model

## 📁 Project Structure

```
IoT-Intrusion-Detection/
│
├── Data/
│   └── RT_IOT2022.csv              # Dataset file
│
├── Notebooks/
│   ├── XGBoostModel.ipynb          # XGBoost implementation
│   ├── RandomForestModel.ipynb     # Random Forest implementation
│   ├── DecisionTreeModel.ipynb     # Decision Tree implementation
│   ├── KNNModel.ipynb              # K-Nearest Neighbors implementation
│   ├── SVMModel.ipynb              # Support Vector Machine implementation
│   ├── NaiveBayesModel.ipynb       # Naive Bayes implementation
│   ├── LogisticRegressionModel.ipynb # Logistic Regression implementation
│   └── TabNetModel.ipynb           # TabNet implementation
│
├── Models/
│   ├── xgboost_model.pkl
│   ├── random_forest_model.pkl
│   ├── decision_tree_model.pkl
│   ├── knn_model.pkl
│   ├── svm_model.pkl
│   ├── naive_bayes_model.pkl
│   ├── logistic_regression_model.pkl
│   ├── tabnet_model.pkl
│   └── *_scaler.pkl / *_label_encoder.pkl
│
├── Results/
│   ├── *_confusion_matrix.png
│   ├── *_feature_importance.png
│   └── model_comparison.png
│
└── README.md
```

## 🔧 Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/IoT-Intrusion-Detection.git
cd IoT-Intrusion-Detection
```

2. **Create a virtual environment** (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate  # On Windows
```

3. **Install required packages**
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Running Individual Models

1. **Navigate to the Notebooks directory**
```bash
cd Notebooks
```

2. **Launch Jupyter Notebook**
```bash
jupyter notebook
```

3. **Open and run any model notebook**
   - XGBoostModel.ipynb
   - RandomForestModel.ipynb
   - DecisionTreeModel.ipynb
   - KNNModel.ipynb
   - SVMModel.ipynb
   - NaiveBayesModel.ipynb
   - LogisticRegressionModel.ipynb
   - TabNetModel.ipynb

### Training Models

Each notebook follows the same structure:
1. Load and explore the dataset
2. Preprocess data (scaling, encoding)
3. Split data into train/test sets
4. Train the model
5. Evaluate performance
6. Visualize results
7. Save the trained model

## 📈 Results

Each model generates:
- **Confusion Matrix**: Visual representation of classification performance
- **Classification Report**: Precision, Recall, F1-Score for each class
- **Accuracy Score**: Overall model accuracy
- **Feature Importance**: Most influential features (where applicable)

Results are saved in the `Results/` directory.

### Model Comparison Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Training Time
- Inference Time

## 📦 Requirements

```
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
jupyter
joblib
pytorch-tabnet
torch
```

Install all requirements:
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter joblib pytorch-tabnet torch
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- Your Name - Initial work

## 🙏 Acknowledgments

- RT-IoT2022 dataset providers
- Scikit-learn documentation
- XGBoost documentation
- PyTorch TabNet developers
- Open-source community

## 📧 Contact

For questions or feedback, please contact: your.email@example.com

---

**Note**: Make sure to download the RT-IoT2022 dataset and place it in the `Data/` directory before running the notebooks.