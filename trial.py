from flask import Flask, request, jsonify
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
import sklearn
import pickle
from flask_cors import CORS
from sklearn.preprocessing import OneHotEncoder
from flask_mail import Mail, Message

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
mail = Mail(app)

app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = 'noreplymentalio@gmail.com'
app.config['MAIL_PASSWORD'] = 'ngcxrdcufoeihirg'
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
mail = Mail(app)

# Load the pre-trained model
with open(r"E:\6.My Project\8th Sem Prj\Flask\Final_Model_DTC.pkl", "rb") as f:
    model = pickle.load(f)

# Load the one-hot encoder for the categorical columns
with open(r"E:\6.My Project\8th Sem Prj\Flask\Final_Encoder.pkl", "rb") as f:
    encoder = pickle.load(f)


@app.route("/", methods=["POST"])
def predict():
    # Extract the form data from the POST request
    data = request.get_json()
    name = data['name']
    email = data['email']
    age = data['age']
    gender = data['gender']

    data.pop("name")
    data.pop("email")
    data.pop("age")
    data.pop("country")
    data.pop("gender")

    # Convert the form data into a DataFrame
    data = pd.DataFrame(data, index=[0])

    # Encode categorical columns
    data['work_interfere'] = {"never": 0, "rarely": 1, "sometimes": 2,"often": 3, "don't Know": 4}[data['work_interfere'].iloc[0]]
    data['family_history'] = {"no": 0, "yes": 1}[data['family_history'].iloc[0]]
    data['care_options'] = {"no": 0, "not sure": 1, "yes": 2, "don't Know": 3}[data['care_options'].iloc[0]]
    data['benefits'] = {"no": 0, "not sure": 1, "yes": 2,"don't Know": 3}[data['benefits'].iloc[0]]
    data['obs_consequence'] = {"no": 0, "yes": 1}[data['obs_consequence'].iloc[0]]
    data['anonymity'] = {"no": 0, "not sure": 1, "yes": 2,"don't Know": 3}[data['anonymity'].iloc[0]]
    data['mental_health_interview'] = {"no": 0, "maybe": 1, "yes": 2, "don't Know": 3}[data['mental_health_interview'].iloc[0]]
    data['wellness_program'] = {"no": 0, "not sure": 1, "yes": 2, "don't Know": 3}[data['wellness_program'].iloc[0]]
    data['seek_help'] = {"no": 0, "not sure": 1, "yes": 2, "don't Know": 3}[data['seek_help'].iloc[0]]

    # Make a prediction using the pre-trained model
    prediction = model.predict(data)[0]

    if prediction == 0:
        prediction = "No"
        state = "Good"
    else:
        prediction = "Yes"
        state = "Bad"

    msg = Message('Hello', sender='mentaleo2023@gmail.com', recipients=[email])
    msg.body = "Subject: Mental Health Prediction Result\n\nDear {name},\n\nAge:{age}, Gender{gender},\n\nYour predicted mental is {state}.\n\nYou need mental health assistance?{prediction}"
    mail.send(msg)
    return jsonify({"prediction": prediction})

    # # Create the message
    # msg = f"Subject: Mental Health Prediction Result\n\nDear {name},\n\nAge:{age}, Gender{gender},\n\nYour predicted mental is {state}.\n\nYou need mental health assistance?{prediction}"
    # # Return the predicted target as a JSON response
    # return jsonify({"prediction": prediction})


if __name__ == '__main__':
    app.run(debug=True)
