# Airbnb Price Prediction Project

This project focuses on predicting Airbnb prices using machine learning models after performing comprehensive data cleaning and feature engineering on the raw Airbnb listings dataset. The dataset includes various attributes about Airbnb properties, such as price, location, amenities, host information, and reviews. 

## Project Goals

1. **Data Cleaning and Preparation:** Handling missing values, correcting inconsistencies, and transforming raw data into a clean, usable format for analysis.
2. **Feature Engineering:** Creating new features from the existing data to improve model performance.
3. **Price Prediction:** Building and evaluating machine learning models to predict the prices of Airbnb listings.

## Project Structure

- **`airbnb-listings.xlsx`**: Raw Airbnb dataset containing information about listings.
- **`airbnb_price_prediction.ipynb`**: Jupyter Notebook containing data cleaning, preprocessing, feature engineering, model training, and evaluation steps.

## Steps Performed

### Data Cleaning & Feature Engineering

1. **Import necessary libraries:** pandas, NumPy, matplotlib, seaborn, etc.
2. **Load the data:** Load the dataset `airbnb-listings.xlsx` into a pandas DataFrame.
3. **Clean the data:** Handle missing values, standardize data formats, and correct inconsistencies in the following columns:
   - Price, Experiences Offered, Host Since, Host Response Time, Host Response Rate, Property Type, Room Type, and more.
4. **Feature Extraction:** Create new features from existing ones, such as splitting out amenities, availability details, review scores, and cancellation policies.
5. **Handle missing data:** Use appropriate imputation methods (e.g., filling with mean/mode or using logical assumptions) based on column relevance.

### Price Prediction

1. **Data Preprocessing:**
   - Import additional libraries such as scikit-learn for model building.
   - Load the dataset `USADATA.xlsx` and drop irrelevant columns.
   - Apply one-hot encoding for categorical variables (e.g., room type, property type).
   - Split the data into training and validation sets (e.g., 80% training, 20% validation).
   - Standardize numerical features to scale them appropriately for model training.
   
2. **Model Training and Evaluation:**
   - Train multiple machine learning models:
     - Linear Regression
     - Ridge Regression
     - Lasso Regression
     - Random Forest Regression
   - Evaluate model performance using metrics like R-squared, adjusted R-squared, and mean squared error (MSE).
   - Visualize results using scatter plots and histograms for predicted vs. actual prices.

3. **Correlation Analysis:**
   - Perform correlation analysis to understand relationships between features and price.
   - Visualize correlations using heatmaps and bar plots.

## Results

The results of the machine learning models, including R-squared, adjusted R-squared, and MSE, are provided in the Jupyter Notebook. The project includes visualizations comparing model performance and feature importance, helping understand which features most impact Airbnb prices.

## Dependencies

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- tqdm

## Usage

1. Clone this repository to your local machine.
2. Ensure all necessary dependencies are installed using `pip install -r requirements.txt` (create this based on the listed dependencies).
3. Replace the dataset path (`airbnb-listings.xlsx` or `USADATA.xlsx`) with your actual dataset.
4. Execute the Jupyter Notebook in a local environment or Google Colab to run the data cleaning, feature engineering, and model training steps.

