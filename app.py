from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load both models
linear_model = pickle.load(open("linear.pkl", "rb"))
rf_model = pickle.load(open("rf.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input values
        hours = float(request.form["hours"])
        attendance = float(request.form["attendance"])
        previous = float(request.form["previous"])
        sleep = float(request.form["sleep"])
        model_choice = request.form["model"]

        # Choose model
        if model_choice == "linear":
            prediction = linear_model.predict([[hours, attendance, previous, sleep]])
            model_used = "Linear Regression"
        else:
            prediction = rf_model.predict([[hours, attendance, previous, sleep]])
            model_used = "Random Forest"

        # Clamp prediction (0–100 realistic range)
        pred_value = max(0, min(100, prediction[0]))

        return render_template(
            "index.html",
            prediction=round(pred_value, 2),
            model_used=model_used
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction="Invalid input! Please enter valid numbers."
        )

if __name__ == "__main__":
    app.run(debug=True)