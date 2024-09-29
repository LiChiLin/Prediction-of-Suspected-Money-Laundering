# Prediction of Suspected Money Laundering

**Authors**: Chi-Lin Li, Wei-Ru Chen, Shih-Chih Lin, Tsung-Han Liu  
**Institution**: National Tsing Hua University, Hsinchu, Taiwan R.O.C.

## Project Overview

This project focuses on developing a machine learning model to detect potential money laundering activities using anonymized customer account transaction data from **E.SUN Bank**. The goal is to identify suspicious transactions while addressing the high false-positive rate present in traditional rule-based systems. The project uses **LightGBM**, a supervised learning algorithm, to predict the likelihood of money laundering and evaluates the model's performance based on precision, recall, and F1-score metrics.

### Key Features:
- **Feature Engineering**: Creation of new features based on customer information and time series transaction data.
- **Handling Imbalanced Data**: Addressing the highly imbalanced nature of the dataset through sampling methods and model adjustments.
- **Machine Learning Models**: Implementation of **Random Forest**, **XGBoost**, and **LightGBM** for prediction, with a focus on optimizing model performance.
- **Scenario-based Modeling**: Three distinct modeling scenarios to evaluate performance based on different data combinations and transformations.

## Methodologies

### 1. Data Preprocessing
- **Customer-Based Preprocessing**: Merging customer information (such as occupation, age, total assets) with alert data to form a baseline model.
- **Transaction-Based Preprocessing**: Merging transaction records (loans, consumption, foreign exchange) with customer data to capture financial behavior patterns over time.
- **Handling Missing Values**: Numerical columns are filled with median values, and categorical columns are filled with mode values. Outliers are removed based on boxplot analysis.

### 2. Feature Engineering
- **Time Series Feature Extraction**: Statistical features (mean, standard deviation, and counts) are extracted from transaction data, and combined with customer data to create enriched feature sets.
- **Missing Value Imputation**: Numerical values are imputed with the median, and categorical data is imputed with the mode.

### 3. Modeling
- **Random Forest**: A baseline ensemble method using decision trees.
- **XGBoost and LightGBM**: Gradient boosting algorithms used to predict suspicious transactions. Both models are optimized using hyperparameter tuning.
- **Handling Imbalanced Data**: Models were adjusted using imbalance functions to handle the significant class imbalance in the dataset.

## Experiment and Results

### Scenario 1: Customer Information Based Model
- Used attributes from the customer information table to build the model.
- **Best Model**: LightGBM  
  - **Precision**: 0.80  
  - **Recall**: 0.96  
  - **F1-Score**: 0.87

### Scenario 2: Customer Information with Time Series Features
- Extended the customer information data by including statistical features from transaction data.
- **Best Model**: LightGBM  
  - **Precision**: 0.96  
  - **Recall**: 1.00  
  - **F1-Score**: 0.98

### Scenario 3: Time Series Transaction Data Merged with Customer Information
- Merged time series transaction data directly with customer information to train the models.
- **Best Model**: LightGBM  
  - **Precision**: 0.99  
  - **Recall**: 1.00  
  - **F1-Score**: 0.99

### Overall Performance
LightGBM outperformed other models across all scenarios. The model was able to identify suspicious money laundering transactions with high precision and recall, achieving an **F1-score of 0.99** in Scenario 3. However, overfitting was observed in some cases due to the imbalanced nature of the dataset.

## Discussion and Future Plans

### Challenges Faced:
- **Imbalanced Data**: The dataset was highly imbalanced, with a class ratio of approximately 1:101. This made it difficult to train models effectively without overfitting to the majority class.
- **Feature Encoding**: An attempt to use **LSTM-autoencoder** for time series data failed to produce meaningful patterns, likely due to the lack of discernible routine patterns in normal transactions.

### Future Plans:
- Explore additional techniques to handle imbalanced datasets, such as **SMOTE** or **cost-sensitive learning**.
- Improve feature encoding techniques, such as exploring advanced time series models for better data representation.

## Installation

To run this project, install the following dependencies:

```bash
pip install numpy pandas scikit-learn xgboost lightgbm matplotlib
```

## Usage

1. **Preprocess Data**: Use the provided scripts to clean and merge customer and transaction data.
2. **Train Models**: Train Random Forest, XGBoost, and LightGBM models on the preprocessed data.
3. **Evaluate Performance**: Use metrics such as precision, recall, and F1-score to evaluate model performance across different scenarios.
4. **Optimize**: Fine-tune the models using hyperparameter optimization to improve performance on imbalanced data.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contribution

Chi-Lin Li, Wei-Ru Chen, Shih-Chih Lin, and Tsung-Han Liu contributed equally to this project.
