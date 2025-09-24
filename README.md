# Loan_Default_Prediction_version_2

# Loan Default Prediction Model

## Overview

This repository contains a Python project developed for a Deep Learning/AI programming course, aimed at predicting loan defaults using historical data from Lending Club (2007-2015). The model addresses a highly imbalanced dataset (84% non-defaults, 16% defaults) to assist financial institutions in identifying risky loans. Leveraging deep learning with Keras/TensorFlow, the code includes data preprocessing, exploratory analysis, feature engineering, and model training, achieving 79% accuracy and a 0.35 F1-score for defaults.

## Features

- **Data Preprocessing**: Transforms categorical data (e.g., loan purpose) into numerical format using one-hot encoding.
- **Exploratory Data Analysis (EDA)**: Visualizes feature distributions and trends (e.g., FICO vs. default status).
- **Feature Engineering**: Removes correlated features (e.g., interest rate) and scales data, with SMOTE for imbalance handling.
- **Deep Learning Model**: Implements a neural network with 64, 32, 16 neurons, L2 regularization, and dropout, optimized with a precision-recall threshold.
- **Evaluation**: Provides classification metrics (F1-score, accuracy, ROC-AUC) and saves predictions.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/loan-default-prediction.git
   cd loan-default-prediction
   ```
2. Install dependencies:
   ```bash
   pip install pandas numpy seaborn matplotlib scikit-learn imbalanced-learn tensorflow
   ```
3. Ensure the dataset `loan_data.csv` is in the project directory (available from course materials or Lending Club datasets).

## Usage

Run the script:
```bash
python Barren_LendingClub_CEP2.py
```
- The script loads data, performs analysis, trains the model, and saves predictions to `output.csv`.
- View plots (histograms, heatmap, training history) during execution.

## Results

- **Accuracy**: 79%
- **Weighted F1-Score**: 0.79
- **Default F1-Score**: 0.35
- **Default Recall**: 36%
- **ROC-AUC**: 0.68

The model captures a notable portion of defaults, though further improvements (e.g., feature transformations) could enhance minority class performance.

## Project Structure

- `Barren_LendingClub_CEP2.py`: Main Python script for the project.
- `loan_data.csv`: Input dataset (replace with your file).
- `transformed_loan_data.csv`: Preprocessed data (generated).
- `output.csv`: Model predictions (generated).

## Improvements Made

- **Initial Model**: Used class weights, achieving 70% accuracy but low default F1 (0.33).
- **Dropout and Threshold**: Added dropout (0.3) and lowered threshold (0.3) for better recall (0.95), though precision dropped (0.18).
- **SMOTE**: Balanced data with synthetic samples, improving recall (0.85) and accuracy (0.48).
- **Regularization**: Introduced L2 (0.01) and tuned learning rate (0.0005), reducing overfitting and boosting accuracy (0.73).
- **Architecture**: Increased layers/neurons (64, 32, 16), enhancing non-default F1 (0.87) and accuracy (0.79).
- **Threshold Optimization**: Used precision-recall curve for optimal threshold (0.48), balancing performance.

## Future Work

- Explore log-transforming skewed features (e.g., `revol.bal`).
- Test advanced loss functions (e.g., focal loss) for better default prediction.
- Investigate additional feature correlations (e.g., `revol.util`).

## License

MIT License - Feel free to use and modify, with attribution.

## Acknowledgements

Thanks to my course instructors and xAI's Grok for guidance in developing this project.

---

*Last updated: September 24, 2025*
