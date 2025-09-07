import pandas as pd
import numpy as np
from pathlib import Path
import json

BASE = Path(__file__).resolve().parent
df = pd.read_csv(BASE / "data" / "housing.csv")
df['age'] = 2025 - df['year_built']
city_dummies = pd.get_dummies(df['city'], prefix='city', drop_first=True)
X = pd.concat([city_dummies, df[['bedrooms','bathrooms','sqft','age','floor']]], axis=1)
y = df['price'].values.reshape(-1,1)
X_mat = np.hstack([np.ones((X.shape[0],1)), X.values])
coef, *_ = np.linalg.lstsq(X_mat, y, rcond=None)
coef = coef.flatten().tolist()
model_spec = {"features": ["INTERCEPT", "city_Bengaluru", "city_Chennai", "city_Delhi", "city_Hyderabad", "city_Mumbai", "city_Pune", "bedrooms", "bathrooms", "sqft", "age", "floor"], "coefficients": [-10285536.006157668, 9263372.688241236, 4255761.848033968, 14824487.239536751, 5146255.583218538, 20571012.092062347, 3360964.763521868, 137415.80647340335, 39979.958288075075, 12550.670497323996, -2004.4374431604488, 9477.244768510549], "description": "Linear regression model saved as coefficients. Prediction = intercept + sum(coef_i * feature_i)", "version": "1.0"}
Path(BASE / "model").mkdir(parents=True, exist_ok=True)
Path(BASE / "model" / "model.json").write_text(json.dumps(model_spec, indent=2))
print('Model re-trained and saved to model/model.json')
