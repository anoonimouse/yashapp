from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# ------------------------------
# Load the trained model
# ------------------------------
try:
    model = joblib.load("model.pkl")
    model_columns = joblib.load("model_columns.pkl")
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Model not found or failed to load:", e)
    model = None
    model_columns = []

# ------------------------------
# Routes
# ------------------------------
@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_text = None

    if request.method == 'POST' and model is not None:
        try:
            # Get input from form
            input_data = {
                "APOE": float(request.form["APOE"]),
                "Clusterin": float(request.form["Clusterin"]),
                "Tau": float(request.form["Tau"]),
                "ABeta42": float(request.form["ABeta42"])
            }

            # Convert to DataFrame
            df = pd.DataFrame([input_data])
            df = df.reindex(columns=model_columns, fill_value=0)

            # Predict
            pred = model.predict(df)[0]
            prediction_text = "✅ Likely Alzheimer's (AD)" if pred == 1 else "🧠 Control (No AD)"

        except Exception as e:
            prediction_text = f"❌ Error: {e}"

    return render_template('index.html', prediction=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
