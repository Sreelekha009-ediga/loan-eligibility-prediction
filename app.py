from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load model and encoders once
MODEL_PATH = os.path.join('model', 'loan_model.pkl')
ENCODERS_PATH = os.path.join('model', 'label_encoders.pkl')

model = joblib.load(MODEL_PATH)
label_encoders = joblib.load(ENCODERS_PATH)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    confidence = None
    eligible = False

    if request.method == 'POST':
        # Get form data
        data = {
            'Gender': request.form['gender'],
            'Married': request.form['married'],
            'Dependents': request.form['dependents'],
            'Education': request.form['education'],
            'Self_Employed': request.form['self_employed'],
            'ApplicantIncome': float(request.form['applicant_income']),
            'CoapplicantIncome': float(request.form['coapplicant_income']),
            'LoanAmount': float(request.form['loan_amount']),
            'Loan_Amount_Term': float(request.form['loan_amount_term']),
            'Credit_History': float(request.form['credit_history']),
            'Property_Area': request.form['property_area']
        }

        # Encode categorical features
        input_df = pd.DataFrame([data])
        for col in ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']:
            input_df[col] = label_encoders[col].transform(input_df[col].astype(str))

        # Predict
        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]  # probability of class 1 (approved)

        eligible = bool(prediction == 1)
        confidence = round(prob * 100, 1) if eligible else round((1 - prob) * 100, 1)
        
        result = "Loan APPROVED!" if eligible else "Loan REJECTED"

    return render_template('index.html', result=result, confidence=confidence, eligible=eligible)

if __name__ == '__main__':
    app.run(debug=True)