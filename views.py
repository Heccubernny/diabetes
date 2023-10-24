from flask import Flask, request, render_template
import pickle
import numpy as np
from preprocessing import MultiLabelEncoder


# views: Blueprint = Blueprint('views', __name__)
app = Flask(__name__)
# Load Pickle Model
encoder = MultiLabelEncoder()

model = pickle.load(open("model.pkl", "rb"))

scaler = pickle.load(open("scaler.pkl", "rb"))


# with open("scaler.pkl", "rb") as f:
#     scaler = pickle.load(f)


# new_user = User(**user_data)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/form")
def base():
    return render_template("form.html")


@app.route("/result", methods=["GET", "POST"])
def predict():
    user_data = {
        "HighBP": float(request.form.get("high-bp")),
        "HighChol": float(request.form.get("high-chol")),
        "CholCheck": float(request.form.get("chol-check")),
        "BMI": float(request.form.get("BMI")),
        "Smoker": float(request.form.get("smoker")),
        "Stroke": float(request.form.get("stroke")),
        "HeartDiseaseorAttack": float(request.form.get("heart-disease-or-attack")),
        "PhysActivity": float(request.form.get("PhysActivity")),
        "Fruits": float(request.form.get("fruits")),
        "Veggies": float(request.form.get("veggies")),
        "HvyAlcoholConsump": float(request.form.get("HvyAlcoholConsump")),
        "AnyHealthcare": float(request.form.get("AnyHealthcare")),
        "NoDocbcCost": float(request.form.get("NoDocbcCost")),
        "GenHlth": float(request.form.get("GenHlth")),
        "MentHlth": float(request.form.get("MentHlth")),
        "PhysHlth": float(request.form.get("PhysHlth")),
        "DiffWalk": float(request.form.get("DiffWalk")),
        "Sex": float(request.form.get("sex")),
        "Age": float(request.form.get("age")),
        "Education": float(request.form.get("education")),
        "Income": float(request.form.get("income")),
    }

    input_data = np.array(list(user_data.values())).reshape(1, -1)

    input_data_encoded = encoder.transform(input_data)

    input_data_scaled = scaler.transform(input_data_encoded)

    prediction = model.predict(input_data_scaled)

    output = prediction
    print(f"here is output {output}, here is my prediction: {prediction}")

    if output == 0:
        output = "Not Diabetes"
    elif output == 1:
        output = "Prediabetes"
    elif output == 2:
        output = "Diabetes"
    else:
        output = "None"

    return render_template("result.html", data=output)


@app.route("/result")
def predictor():
    return render_template("result.html")


if __name__ == "__main__":
    app.run(debug=True)
