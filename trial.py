from flask import Flask, request, render_template
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
import sklearn
import pickle

app = Flask(__name__)

# Load the pre-trained model
# model = pickle.load("Final_Model_DTC.pkl")
with open(r"E:\6.My Project\8th Sem Prj\Flask\Final_Model_DTC.pkl", "rb") as f:
    model = pickle.load(f)
    
@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        # Extract the form data
        work_interfere = request.form.get("work_interfere")
        family_history = request.form.get("family_history")
        care_options = request.form.get("care_options")
        benefits = request.form.get("benefits")
        obs_consequence = request.form.get("obs_consequence")
        anonymity = request.form.get("anonymity")
        mental_health_interview = request.form.get("mental_health_interview")
        wellness_program = request.form.get("wellness_program")
        seek_help = request.form.get("seek_help")

        work_interfere = {"Never": 0, "Rarely": 1, "Sometimes": 2, "Often": 3}[work_interfere]
        family_history = {"No": 0, "Yes": 1}[family_history]
        care_options = {"No": 0, "Not sure": 1, "Yes": 2}[care_options]
        benefits = {"No": 0, "Not sure": 1, "Yes": 2}[benefits]
        obs_consequence = {"No": 0, "Yes": 1}[obs_consequence]
        anonymity = {"No": 0, "Not sure": 1, "Yes": 2}[anonymity]
        mental_health_interview = {"No": 0, "Maybe": 1, "Yes": 2}[mental_health_interview]
        wellness_program = {"No": 0, "Not sure": 1, "Yes": 2}[wellness_program]
        seek_help = {"No": 0, "Not sure": 1, "Yes": 2}[seek_help]

        # Create a DataFrame with the form data
        data = pd.DataFrame({
            "work_interfere": [work_interfere],
            "family_history": [family_history],
            "care_options": [care_options],
            "benefits": [benefits],
            "obs_consequence": [obs_consequence],
            "anonymity": [anonymity],
            "mental_health_interview": [mental_health_interview],
            "wellness_program": [wellness_program],
            "seek_help": [seek_help]
        })

        # Make a prediction using the pre-trained model
        prediction = model.predict(data)[0]

        if prediction == 0:
            prediction = "No"
        else:
            prediction = "Yes"

        # Render the result template with the predicted target
        return render_template("result.html", target=prediction)

    # Render the form template on GET request
    return render_template("index.html")


if __name__ == '__main__':
    app.run(debug=True)
