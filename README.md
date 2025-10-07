

https://github.com/user-attachments/assets/111c6b31-bf4e-4eba-b611-ce0625b1ee23

# 🏡 Smart House Price Prediction Model

A modern Flask + Machine Learning web application that predicts house prices accurately based on multiple property features such as area, rooms, city, house type, bathrooms, parking, furnishing, floor number, total floors, year built, facing direction, and nearby amenities.  
Designed with a clean, responsive, and user-friendly interface for smooth interaction and easy visualization.

-----------------------------------------------------------------------------------------------------------------------------

## 🚀 Features
- Predicts property price instantly using a trained ML model 📊  
- User-friendly web interface built with **Flask, HTML, and CSS** 🌐  
- Supports multiple property details for accurate predictions  
- Clean and responsive UI for better visualization 📱  
- Easy to run locally and understand the prediction flow 💡  

-----------------------------------------------------------------------------------------------------------------------------

## 🛠 Tech Stack
- **Python** 🐍  
- **Flask** 🌶  
- **Scikit-learn / Pandas / NumPy** 📈  
- **HTML5 & CSS3** 🎨  
- **Git & GitHub** 🔗  

-----------------------------------------------------------------------------------------------------------------------------

## ⚙️ How to run
1. Clone the repository:
   git clone https://github.com/atharva18-hue/smart-house-price-prediction-model.git

2. Navigate into the project folder:
   cd smart-house-price-prediction-model

3. Install dependencies:
   pip install -r requirements.txt

4. Run the Flask app:
   python app.py

5. Open the browser at:
   http://127.0.0.1:5000

-----------------------------------------------------------------------------------------------------------------------------

## Project Structure

smart-house-price-prediction-model/
│── app.py                 # Flask application
│── train.py               # Model training script
│── requirements.txt       # Python dependencies
│── data/
│   └── housing.csv        # Dataset
│── model/
│   └── model.json         # Trained model
│── templates/
│   ├── index.html         # Input form page
│   └── result.html        # Prediction result page
│── static/
│   └── styles.css         # CSS styles

-----------------------------------------------------------------------------------------------------------------------------

## Flask Application (app.py)
python
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model/model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = np.array([features])
    prediction = model.predict(final_features)
    return render_template('result.html', prediction_text=f'₹ {prediction[0]:,.2f}')

if __name__ == "__main__":
    app.run(debug=True)

-----------------------------------------------------------------------------------------------------------------------------

## Model Training (train.py)
python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
data = pd.read_csv('data/housing.csv')

# Features and target
X = data[['area', 'rooms', 'bathrooms', 'parking']]
y = data['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
with open('model/model.pkl', 'wb') as f:
    pickle.dump(model, f)

-----------------------------------------------------------------------------------------------------------------------------

## HTML Input Form (templates/index.html)
html
Copy code
<form action="/predict" method="post">
    <label>Area (sq.ft)</label>
    <input type="number" name="area" required>
    <label>Rooms</label>
    <input type="number" name="rooms" required>
    <label>Bathrooms</label>
    <input type="number" name="bathrooms" required>
    <label>Parking</label>
    <select name="parking">
        <option>No Parking</option>
        <option>1 Parking</option>
        <option>2 Parking</option>
    </select>
    <button type="submit">Predict Price</button>
</form>

-----------------------------------------------------------------------------------------------------------------------------

## HTML Result Page (templates/result.html)
This page displays the predicted house price after the user submits the input form.
It shows a clean, highlighted result box with the estimated price in a readable format.
Users can quickly see the output and compare different property inputs by returning to the input page.
The layout is responsive and visually consistent with the main form page, ensuring smooth user interaction.
A Try Next Prediction” button allows users to easily input new data without refreshing the browser.

----------------------------------------------------------------------------------------------------------------------------------

## Code overview
- app.py: Main Flask application handling routes, form submission, and prediction display.
- train.py: Script to train the ML model using dataset from data/housing.csv.
- model/: Folder containing trained model (model.pkl / model.json).
- templates/index.html: Input form page for users to enter property details.
- templates/result.html: Page showing predicted house price.
- static/styles.css: Custom CSS styles for the web app.
  
----------------------------------------------------------------------------------------------------------------------------------

 ## Contributing
Contributions, issues, and feature requests are welcome!
Feel free to fork this repo and submit a PR.

-----------------------------------------------------------------------------------------------------------------------------

## Author
@Atharva Chavhan

-----------------------------------------------------------------------------------------------------------------------------
