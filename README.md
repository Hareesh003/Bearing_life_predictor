# Predictive Maintenance using Machine Learning

## Overview

This project aims to predict the Remaining Useful Life (RUL) of aircraft engines using machine learning techniques. The dataset used is the NASA Turbofan Engine Degradation Simulation Dataset (FD001).

## Data Preprocessing

The following preprocessing steps were applied to the data:

1. **Handling Missing Values:** The dataset was checked for missing values, and no missing values were found.
2. **Feature Scaling:** StandardScaler was used to standardize the numeric features.
3. **Outlier Detection and Removal:** Box plots were used to identify outliers, and outliers were removed using the z-score method.
4. **Feature Selection:** Correlation analysis and domain knowledge were used to select relevant features for model training. Features with low correlation to the target variable and those deemed less relevant were dropped.

## Data Analysis

1. **Descriptive Statistics:** The describe() method was used to calculate descriptive statistics for the dataset.
2. **Correlation Analysis:** A heatmap of the correlation matrix was generated to visualize the relationships between features.
3. **Time Series Analysis:** Time series plots were generated for selected features to explore trends and patterns. Rolling statistics were also calculated to further analyze the time series data.

## Model Training and Evaluation

Three machine learning models were trained and evaluated:

1. **Random Forest Regressor:** A Random Forest model was trained using the selected features and the time_in_cycles as the target variable. The model was evaluated using Mean Squared Error (MSE) and R-squared on the validation set.
2. **Multi-Layer Perceptron (MLP) Regressor:** An MLP model was trained with similar settings as the Random Forest model. MSE and R-squared were used for evaluation on the validation set.
3. **K-Nearest Neighbors (KNN) Regressor:** A KNN model was trained with similar settings as the previous models. MSE and R-squared were used for evaluation on the validation set.

## Hyperparameter Tuning

GridSearchCV was used to tune the hyperparameters of each model. The best parameters for each model were obtained and used for the final evaluation.

## Results

The following table summarizes the results of the analysis:

| Model | Best Parameters | MSE (Validation) | R-squared (Validation) |
|---|---|---|---|
| Random Forest | `estimator__n_estimators`: 100, `estimator__max_depth`: 10 | `rf_mse_val` | `rf_r2_val` |
| MLPRegressor | `hidden_layer_sizes`: (30, 30), `activation`: 'relu' | `ann_mse_val` | `ann_r2_val` |
| KNeighborsRegressor | `n_neighbors`: 3 | `knn_mse_val` | `knn_r2_val` |

**Note:** Replace `rf_mse_val`, `rf_r2_val`, `ann_mse_val`, `ann_r2_val`, `knn_mse_val`, and `knn_r2_val` with the actual values obtained from your analysis.

## Conclusion

Based on the results, the Random Forest model achieved the best performance on the validation set, followed by the MLPRegressor and KNeighborsRegressor. This model can be used to predict the RUL of aircraft engines and potentially help in scheduling maintenance.

## Future Work

- Explore other machine learning models and techniques.
- Further optimize the hyperparameters of the models.
- Develop a web application or API to deploy the model for real-time predictions.
