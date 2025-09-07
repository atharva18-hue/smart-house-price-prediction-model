Smart House Price Predictor (Portable)

What's included:
- data/housing.csv           (synthetic dataset)
- train.py                  (regenerates model.json from CSV)
- model/model.json          (linear model coefficients + feature order)
- app.py                    (Flask app; no sklearn required at runtime)
- templates/index.html
- templates/result.html
- static/styles.css
- requirements.txt

How to run:
1) Create & activate a Python venv:
   python -m venv venv
   .\venv\Scripts\activate     (Windows)  OR  source venv/bin/activate (Mac/Linux)

2) Install requirements:
   pip install -r requirements.txt

3) (Optional) Re-train model from CSV (will overwrite model/model.json):
   python train.py

4) Run the app:
   python app.py

5) Open browser: http://127.0.0.1:5000

Notes:
- The app uses a JSON-stored linear model for portability (avoids sklearn pickle issues).
