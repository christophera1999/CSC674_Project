from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
import sqlite3

# Load the trained model
model = joblib.load('voting_classifier.pkl')

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('web.html')


from sklearn.preprocessing import LabelEncoder

@app.route('/predict', methods=['POST'])
def predict():
    # Collect data from the form
    data = request.form.to_dict()

    # Convert form data to DataFrame
    features = pd.DataFrame([data], columns=[
        'Loan_ID', 'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
        'Credit_History', 'Property_Area', 'age', 'hypertension', 'heart_disease',
        'avg_glucose_level', 'bmi', 'smoking_status', 'stroke'
    ])

    # Preprocess features (add sensitive_feature, drop Loan_ID)
    features['sensitive_feature'] = features['Gender']
    features = features.drop(columns=['Loan_ID'])

    # Encode categorical variables
    categorical_columns = ['Gender', 'Married', 'Education', 'Self_Employed',
                           'Property_Area', 'smoking_status']
    label_encoders = {}
    for column in categorical_columns:
        le = LabelEncoder()
        features[column] = le.fit_transform(features[column])
        label_encoders[column] = le

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    features = pd.DataFrame(imputer.fit_transform(features), columns=features.columns)

    # Predict Loan_Status
    prediction = model.predict(features)
    result = "Approved" if prediction[0] == 1 else "Denied"
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)