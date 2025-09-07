from flask import Flask, render_template, request
import random

app = Flask(__name__)

# Dummy model (random prediction logic)
class DummyModel:
    def predict(self, features):
        area = features[0][0]
        rooms = features[0][1]

        base_price = area * 5000 + rooms * 200000
        variation = random.uniform(-0.2, 0.2) * base_price  # +/- 20% variation
        return [base_price + variation]

model = DummyModel()

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        area = float(request.form['area'])
        rooms = int(request.form['rooms'])
        year = request.form.get('year')
        furnishing = request.form.get('furnishing')
        city = request.form.get('city')

        prediction = model.predict([[area, rooms]])
        output = prediction[0]

        # Convert to Indian style (Lakh / Crore)
        if output >= 10000000:
            price_text = f"₹ {output/10000000:.2f} Crore"
        else:
            price_text = f"₹ {output/100000:.2f} Lakh"

        details = f"Location: {city}, Rooms: {rooms}, Area: {area} sq.ft, Year: {year}, Furnishing: {furnishing}"

        return render_template(
            "result.html",
            prediction_text=f"Predicted Price: {price_text}",
            build_by="Build by Atharva Chavhan",
            details=details
        )
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
