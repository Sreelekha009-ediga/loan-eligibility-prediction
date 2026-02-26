# Loan Eligibility Prediction

**Live Demo:** https://loan-eligibility-prediction-3.onrender.com

A supervised machine learning project to predict loan eligibility based on applicant details.

### Key Highlights (as on resume)
- Developed supervised ML models to predict loan eligibility using structured financial datasets.
- Compared multiple algorithms and achieved high accuracy with **Random Forest (~84.55%)**.
- Performed feature engineering, data preprocessing (handling missing values, label encoding), and model evaluation.
- Built a clean web interface with Flask + HTML/CSS for user-friendly predictions.
- Tech Stack: Python, Pandas, NumPy, Scikit-learn, Flask, HTML/CSS

### Project Structure
loan-eligibility-prediction/
├── app.py                  # Flask backend
├── train.py                # Model training script
├── requirements.txt
├── templates/
│   └── index.html          # Frontend form
├── static/
│   └── style.css           # Styling
├── model/
│   ├── loan_model.pkl
│   └── label_encoders.pkl
└── README.md


### Results
- **Random Forest**: 84.55% accuracy on test set
- Precision/Recall/F1 for approved loans: ~0.84 / 0.95 / 0.90
- Key feature: Credit_History has the strongest impact

### How to Run Locally
1. Clone the repo:
   ```bash
   git clone https://github.com/YOUR-USERNAME/loan-eligibility-prediction.git
   cd loan-eligibility-prediction

2. Install dependencies:pip install -r requirements.txt
3. Train the model (run once): python train.py
4. Start the app: python app.py
5. Open http://127.0.0.1:5000 in your browser


