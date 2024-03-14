# Earthquake Prediction Model README

## Overview:
This repository contains code for building a machine learning model to predict the occurrence of tsunamis following earthquakes. The model is trained on earthquake data from 1995 to 2023.

## Contents:
1. **earthquake_1995-2023.csv**: Dataset containing earthquake data.
2. **earthquake_prediction.ipynb**: Jupyter notebook containing the code for data preprocessing, model building, and evaluation.
3. **README.md**: This file providing an overview of the repository.
4. **eda.ipynb**: jupyter notebook consisting of the Exploratory Data Analysis and visualisations.
5. **confusion_matrix.png**: Image file showing the confusion matrix for the model.

## Requirements:
- Python 3.x
- Jupyter Notebook
- Libraries: numpy, pandas, seaborn, matplotlib, scikit-learn, imbalanced-learn, xgboost

## Instructions:
1. **Clone Repository**: Clone this repository to your local machine using the following command:
   ```
   git clone https://github.com/your_username/earthquake_prediction.git
   ```
2. **Install Dependencies**: Install the required libraries using pip:
   ```
   pip install -r requirements.txt
   ```
3. **Run Jupyter Notebook**: Open the `earthquake_prediction.ipynb` notebook in Jupyter Notebook and execute each cell sequentially.

## Model Evaluation:
- The notebook contains code for training multiple machine learning models including Logistic Regression, Decision Tree, Random Forest, XGBoost, and Naive Bayes.
- Model evaluation metrics such as precision, recall, F1-score, and accuracy are provided for both training and testing sets.
- The Random Forest model consistently performs well across different scenarios and is recommended for prediction.

## Future Improvements:
- Explore more feature engineering techniques to improve model performance.
- Experiment with other machine learning algorithms and hyperparameter tuning for better results.
- Consider incorporating additional data sources for enhanced prediction accuracy.

Feel free to contribute to this project by opening issues, suggesting improvements, or submitting pull requests.
