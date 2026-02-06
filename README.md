# IoT Intrusion Detection System

A machine learning project for detecting intrusions in IoT networks using multiple classification algorithms including XGBoost, Random Forest, Decision Tree, SVM, and TabNet.

## 📊 Dataset

The dataset used in this project can be downloaded from: [https://www.kaggle.com/competitions/machinelearningassignment3/data]

### Data Files
- `train.csv` - Training dataset
- `test_without_label.csv` - Test dataset without labels
- `cleaned_data.csv` - Preprocessed/cleaned dataset

## 📁 Project Structure

```
IoT-Intrusion-Detection/
│
├── Data/
│   ├── train.csv
│   ├── test_without_label.csv
│   └── cleaned_data.csv
│
├── Notebooks/
│   ├── DataPreprocessing.ipynb
│   ├── XGboostModel.ipynb
│   ├── RandomForestModel.ipynb
│   ├── DecisionTreeModel.ipynb
│   ├── SvmModel.ipynb
│   └── TabNetModel.ipynb
│
└── README.md
```

## 🤖 Models Implemented

| Model | Notebook |
|-------|----------|
| XGBoost | `XGboostModel.ipynb` |
| Random Forest | `RandomForestModel.ipynb` |
| Decision Tree | `DecisionTreeModel.ipynb` |
| SVM | `SvmModel.ipynb` |
| TabNet | `TabNetModel.ipynb` |

## 🔧 Installation

```bash
git clone https://github.com/yourusername/IoT-Intrusion-Detection.git
cd IoT-Intrusion-Detection
pip install pandas numpy scikit-learn xgboost matplotlib seaborn pytorch-tabnet torch
```

## 🚀 Usage

1. Download the dataset from the link above and place files in the `Data/` folder
2. Run `DataPreprocessing.ipynb` first to prepare the data
3. Run any model notebook to train and evaluate

```bash
cd Notebooks
jupyter notebook
```

## 📦 Requirements

- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- seaborn
- pytorch-tabnet
- torch

## 📝 License

MIT License