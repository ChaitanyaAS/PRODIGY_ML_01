import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import zipfile
import os

# Define paths
zip_path = r'C:\Users\User\Downloads\house-prices-advanced-regression-techniques.zip'
extract_dir = r'C:\Users\User\Downloads\house_prices'
model_file = 'house_price_model.pkl'

# Extract the ZIP file if the model is not available
if not os.path.exists(model_file):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    # Path to the dataset file
    dataset_path = os.path.join(extract_dir, 'train.csv')

    # Load the dataset
    data = pd.read_csv(dataset_path)

    # Feature selection
    X = data[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
    y = data['SalePrice']

    # Train the model
    model = LinearRegression()
    model.fit(X, y)

    # Save the model
    joblib.dump(model, model_file)
    print("Model trained and saved.")
