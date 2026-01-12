from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

model = joblib.load("models/model.joblib")

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

  
    df = pd.DataFrame([data])


    pred = model.predict(df)[0]

    return jsonify({
        "prediction": int(pred)
    })

if __name__ == "__main__":
    app.run(port=3330, debug=True)
