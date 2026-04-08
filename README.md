# CardioSense — Heart Disease Prediction Web App

## Where is data stored?
`cardiosense.db` — a SQLite file created automatically on first run.
Contains three tables:
  - user         → name, email, hashed password, registration date
  - otp          → temporary 6-digit codes for registration (10-min expiry)
  - prediction   → every prediction: all 11 clinical inputs + model results + risk score

---

## Setup

### 1. Install dependencies
pip install flask flask-sqlalchemy reportlab pandas numpy scikit-learn xgboost werkzeug requests

### 2. Configure EmailJS (real OTP delivery — free)

This app uses EmailJS REST API to send OTP emails FROM vishuusharma479@gmail.com TO the user's email.
No SMTP password needed. Steps:

  a) Go to https://www.emailjs.com → Create free account
  b) Dashboard → Email Services → Add New Service → Gmail
     Connect vishuusharma479@gmail.com → copy the SERVICE_ID
  c) Dashboard → Email Templates → Create New Template
       Subject  : Your CardioSense Verification Code
       To Email : {{to_email}}
       Body     :
         Hi {{to_name}},
         Your one-time verification code is: {{otp_code}}
         This code expires in 10 minutes. Do not share it.
       → Save → copy the TEMPLATE_ID
  d) Dashboard → Account → API Keys → copy PUBLIC_KEY

  e) Open app.py and update lines 30–32:
       EMAILJS_SERVICE_ID  = 'service_xxxxxx'
       EMAILJS_TEMPLATE_ID = 'template_xxxxxx'
       EMAILJS_PUBLIC_KEY  = 'xxxxxxxxxxxxxxxxx'

If EmailJS is NOT configured: app runs in dev mode — OTP is printed to terminal console.

### 3. Train models (to achieve 93-95%+ accuracy)
  a) Update heart.csv path in ip-2-updated.py (line 14)
  b) Run:  python ip-2-updated.py
  c) Copy all .pkl files + scaler.pkl into the cardiosense/ folder

If .pkl files are missing: app runs demo mode — predictions are simulated.

### 4. Run
python app.py
Open: http://localhost:5000

---

## Auth Flow
  LOGIN    → Email + Password (no OTP)
  REGISTER → Fill details → OTP sent to user's email → verify → account created

---

## Accuracy
  Original ip-2.py results:
    Random Forest      88.58%
    Logistic Regression 86.41%
    Gradient Boosting   85.86%
    XGBoost             85.32%
    SVM                 83.69%
    Decision Trees      80.97%

  With ip-2-updated.py (feature engineering + SMOTE + better GridSearch):
    Expected: 93–96% for top models
