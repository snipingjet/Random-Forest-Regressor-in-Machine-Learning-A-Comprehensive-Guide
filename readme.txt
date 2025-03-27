# Housing Price Prediction using Random Forest Regressor

Welcome to the **Housing Price Prediction** project! This repository demonstrates how to build a **Random Forest Regressor (RFR)** model to predict housing prices based on various features such as median income, house age, and location.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation Instructions](#installation-instructions)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Evaluation](#model-evaluation)
- [Visualizations](#visualizations)
- [Conclusion](#conclusion)
- [License](#license)

## Project Overview

In this project, we utilize the **Random Forest Regressor (RFR)** to predict housing prices. The model is trained using a dataset that includes various features like:
- **Median Income**
- **House Age**
- **Latitude & Longitude**
- **Average Occupants per Household**

The goal is to predict housing prices accurately based on these features. The model's performance is evaluated using **Mean Squared Error (MSE)**, **Root Mean Squared Error (RMSE)**, and **R-squared (R²)** metrics. Additionally, we analyze the **feature importance** to understand which variables most impact housing price predictions.

## Dataset

The dataset used for this project contains features about housing in California. These include:
- **MedInc (Median Income)**
- **HouseAge (Median Age of Houses)**
- **AveRooms (Average Rooms per Household)**
- **AveOccup (Average Occupants per Household)**
- **Longitude & Latitude**

The target variable is the **housing price**.

## Installation Instructions

To run this project locally, you'll need to set up your environment by installing the required dependencies.

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/yourusername/housing-price-rfr.git
   ```

2. Navigate to the project directory:
   ```bash
   cd housing-price-rfr
   ```

3. Install the necessary Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

   This will install the following libraries:
   - pandas
   - numpy
   - scikit-learn
   - matplotlib
   - seaborn

## Usage

After setting up your environment, you can execute the Python script to train the **Random Forest Regressor** model, evaluate its performance, and visualize the feature importances.

1. Run the script:
   ```bash
   python housing_price_rfr.py
   ```

   The script will output:
   - **Model evaluation metrics** such as **Mean Squared Error (MSE)**, **Root Mean Squared Error (RMSE)**, and **R-squared (R²)**.
   - **Feature importance visualization** showing which features contribute most to the model's predictions.
   - A plot of **actual vs predicted values**.

## Project Structure

The project directory contains the following files:

- **housing_price_rfr.py**: Python script that loads the dataset, trains the RFR model, evaluates performance, and visualizes results.
- **requirements.txt**: A list of required dependencies.
- **feature_importance.png**: A plot showing the importance of each feature used in the model.
- **actual_vs_predicted.png**: A plot showing the relationship between actual and predicted housing prices.

## Model Evaluation

The model’s performance is evaluated using the following metrics:

1. **Mean Squared Error (MSE)**: Measures the average squared differences between predicted and actual values. A lower value indicates better model performance.
2. **Root Mean Squared Error (RMSE)**: The square root of MSE, giving a more interpretable error measure in the same units as the target variable.
3. **R-squared (R²)**: Represents the proportion of variance in the target variable explained by the model. A higher value indicates better model fit.

## Visualizations

1. **Feature Importance**:
   - This bar plot shows the relative importance of each feature in making predictions.

   ![Feature Importance](feature_importance.png)

2. **Actual vs Predicted Values**:
   - A scatter plot comparing actual housing prices with predicted values.

   ![Actual vs Predicted](actual_vs_predicted.png)

## Conclusion

The **Random Forest Regressor (RFR)** model performed effectively, achieving a **R² of 0.81**, meaning it explained 81% of the variance in the target variable (housing price). The low **Mean Squared Error (MSE)** and **Root Mean Squared Error (RMSE)** values suggest that the model makes accurate predictions.

### Future Improvements:
- Tune the model’s hyperparameters for better performance.
- Compare the results with other algorithms like **Gradient Boosting** or **XGBoost**.
- Consider feature engineering or removing less impactful features to simplify the model.

Feel free to explore and adapt this code for your own projects or tutorials. Contributions are welcome!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---