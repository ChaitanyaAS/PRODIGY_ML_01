from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

data = pd.DataFrame({
    'GrLivArea': [1000, 1500, 2000, 2500, 3000],
    'BedroomAbvGr': [2, 3, 3, 4, 4],
    'FullBath': [1, 2, 2, 3, 3],
    'SalePrice': [200000, 250000, 300000, 350000, 400000]
})

X = data[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
y = data['SalePrice']

model = LinearRegression()
model.fit(X, y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    square_footage = float(request.form['square_footage'])
    bedrooms = int(request.form['bedrooms'])
    bathrooms = int(request.form['bathrooms'])

    features = np.array([[square_footage, bedrooms, bathrooms]])
    prediction = model.predict(features)[0]

    return render_template('result.html', price=prediction)

if __name__ == '__main__':
    app.run(debug=True)
