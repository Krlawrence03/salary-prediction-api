from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load your trained model (same name from Step 1)
model = joblib.load("salary_predict_model.ml")

# Expected input fields — must match the model training columns
EXPECTED_COLS = [
    "age",
    "gender",
    "country",
    "highest_deg",
    "code_experience",
    "current_title",
    "company_size"
]

@app.route("/")
def home():
    return "<h1>Salary Prediction API</h1><p>Built by Kyle Lawrence</p>"

@app.route("/predict", methods=["POST"])
def predict():
    # did we actually get JSON?
    if not request.is_json:
        return jsonify({"error": "Request must be JSON with Content-Type: application/json"}), 415

    data = request.get_json()

    # check if we got all the keys we need
    missing = [col for col in EXPECTED_COLS if col not in data]
    if missing:
        return jsonify({"error": f"Missing keys: {missing}", "got": data}), 400

    # build dataframe in correct order
    row = [[data[col] for col in EXPECTED_COLS]]
    df = pd.DataFrame(row, columns=EXPECTED_COLS)

    pred = model.predict(df)[0]
    return jsonify({"prediction": float(pred)})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5003, debug=True)