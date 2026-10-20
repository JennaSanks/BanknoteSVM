
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the trained model
MODEL_PATH = Path(__file__).resolve().parent / "banknote_svm.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Model file not found. Run train_model.py first."
    )

model = joblib.load(MODEL_PATH)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    values = {
        "variance": "",
        "skewness": "",
        "curtosis": "",
        "entropy": ""
    }

    if request.method == "POST":
        try:
            # Get values from the HTML form
            for key in values:
                values[key] = request.form.get(key, "").strip()

            features = np.array([[
                float(values["variance"]),
                float(values["skewness"]),
                float(values["curtosis"]),
                float(values["entropy"])
            ]])

            if not np.isfinite(features).all():
                raise ValueError("Values must be finite numbers.")

            # Predict the banknote class
            prediction = int(model.predict(features)[0])

            if prediction == 0:
                result = "Genuine Banknote"
            else:
                result = "Counterfeit Banknote"

        except (ValueError, TypeError):
            error = "Please enter valid numerical values in all fields."

    return render_template(
        "index.html",
        result=result,
        error=error,
        values=values
    )


if __name__ == "__main__":
    app.run(debug=True)
