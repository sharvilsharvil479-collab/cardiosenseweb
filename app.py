"""
CardioSense — Heart Disease Prediction Web App
OTP via EmailJS REST API (no SMTP setup required)
"""
import os, json, random, string, pickle, io, requests as req_lib
from datetime import datetime, timedelta
from functools import wraps
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from flask import (Flask, render_template, request, redirect,
                   url_for, session, jsonify, send_file, abort)
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, HRFlowable)
from reportlab.lib.enums import TA_CENTER

# ─────────────────────────────────────────────────────────────────────
# App Config
# ─────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'cardiosense_secret_key_2024')

# Use PostgreSQL from Vercel env variable if available, else local SQLite
db_url = os.environ.get('POSTGRES_URL', os.environ.get('DATABASE_URL', 'sqlite:///cardiosense.db'))
# Standardize Postgres dialect for SQLAlchemy
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = db_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)

# ─────────────────────────────────────────────────────────────────────
# EmailJS Configuration
# HOW TO SETUP (free, 200 emails/month):
#   1. Go to https://www.emailjs.com and create a free account
#   2. Add Email Service → connect vishuusharma479@gmail.com
#      (Dashboard → Email Services → Add New Service → Gmail)
#   3. Copy your SERVICE_ID from the service page
#   4. Create an Email Template:
#      Dashboard → Email Templates → Create New Template
#      Subject: "Your CardioSense Verification Code"
#      Body:  Hello {{to_name}}, Your OTP is: {{otp_code}}  (expires in 10 min)
#      To Email: {{to_email}}
#   5. Copy TEMPLATE_ID and PUBLIC_KEY (Account → API Keys)
#   6. Fill in the 3 values below
# ─────────────────────────────────────────────────────────────────────
EMAILJS_SERVICE_ID  = os.environ.get('EMAILJS_SERVICE_ID', 'YOUR_SERVICE_ID')
EMAILJS_TEMPLATE_ID = os.environ.get('EMAILJS_TEMPLATE_ID', 'YOUR_TEMPLATE_ID')
EMAILJS_PUBLIC_KEY  = os.environ.get('EMAILJS_PUBLIC_KEY', 'YOUR_PUBLIC_KEY')

db = SQLAlchemy(app)

# ─────────────────────────────────────────────────────────────────────
# Database Models
# ─────────────────────────────────────────────────────────────────────
class User(db.Model):
    id            = db.Column(db.Integer, primary_key=True)
    name          = db.Column(db.String(120), nullable=False)
    email         = db.Column(db.String(200), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)
    predictions   = db.relationship('Prediction', backref='user', lazy=True)

class OTP(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    email      = db.Column(db.String(200), nullable=False)
    code       = db.Column(db.String(6), nullable=False)
    expires_at = db.Column(db.DateTime, nullable=False)
    used       = db.Column(db.Boolean, default=False)

class Prediction(db.Model):
    id           = db.Column(db.Integer, primary_key=True)
    user_id      = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at   = db.Column(db.DateTime, default=datetime.utcnow)
    age          = db.Column(db.Integer)
    sex          = db.Column(db.Integer)
    chestpain    = db.Column(db.Integer)
    rbp          = db.Column(db.Integer)
    cholesterol  = db.Column(db.Integer)
    fasting_bs   = db.Column(db.Integer)
    ecg          = db.Column(db.Integer)
    max_hr       = db.Column(db.Integer)
    angina       = db.Column(db.Integer)
    oldpeak      = db.Column(db.Float)
    st_slope     = db.Column(db.Integer)
    risk_pct     = db.Column(db.Integer)
    risk_label   = db.Column(db.String(20))
    results_json = db.Column(db.Text)

with app.app_context():
    db.create_all()

# ─────────────────────────────────────────────────────────────────────
# Jinja Filter
# ─────────────────────────────────────────────────────────────────────
@app.template_filter('from_json')
def from_json_filter(value):
    if not value:
        return []
    try:
        return json.loads(value)
    except Exception:
        return []

# ─────────────────────────────────────────────────────────────────────
# OTP via EmailJS REST API
# ─────────────────────────────────────────────────────────────────────
def gen_otp():
    return ''.join(random.choices(string.digits, k=6))

def send_otp_via_emailjs(to_email, to_name, otp_code):
    """
    Sends OTP email via EmailJS REST API.
    From: vishuusharma479@gmail.com (your EmailJS sender)
    To:   the user's email address
    Returns: (True, None) on success or (False, error_message) on failure
    """
    # Check if credentials are configured
    if (EMAILJS_SERVICE_ID  == 'YOUR_SERVICE_ID' or
        EMAILJS_TEMPLATE_ID == 'YOUR_TEMPLATE_ID' or
        EMAILJS_PUBLIC_KEY  == 'YOUR_PUBLIC_KEY'):
        # Not configured — print to console for development
        print(f"\n{'='*55}")
        print(f"  [DEV MODE] OTP for {to_email}: {otp_code}")
        print(f"  Configure EmailJS keys in app.py to send real emails.")
        print(f"{'='*55}\n")
        return False, 'emailjs_not_configured'

    url     = 'https://api.emailjs.com/api/v1.0/email/send'
    payload = {
    'service_id': EMAILJS_SERVICE_ID,
    'template_id': EMAILJS_TEMPLATE_ID,

    # ✅ ADD BOTH (IMPORTANT)
    'user_id': EMAILJS_PUBLIC_KEY,

    'template_params': {
        'to_name': to_name,
        'to_email': to_email,
        'otp_code': otp_code,
        'from_name': 'CardioSense',
    }
}
    try:
        resp = req_lib.post(url, json=payload, timeout=10)
        print("\n====== EMAILJS DEBUG ======")
        print("Payload:", payload)
        print("Status:", resp.status_code)
        print("Response:", resp.text)
        print("JSON:", resp.json())
        print("===========================\n")
        if resp.status_code == 200:
            app.logger.info(f"OTP sent successfully to {to_email}")
            return True, None
        else:
            app.logger.error(f"EmailJS error {resp.status_code}: {resp.text}")
            return False, f'EmailJS API error: {resp.status_code}'
    except req_lib.exceptions.Timeout:
        return False, 'Request timed out. Check your internet connection.'
    except req_lib.exceptions.RequestException as e:
        return False, str(e)

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

# ─────────────────────────────────────────────────────────────────────
# ML Models
# ─────────────────────────────────────────────────────────────────────
ML_MODELS = {
    'Decision Trees':      'DecisionTreeClassifier.pkl',
    'Logistic Regression': 'LogisticRegression.pkl',
    'Random Forest':       'RandomForestClassifier.pkl',
    'SVM':                 'svm.pkl',
    'XGBoost':             'XGBClassifier.pkl',
    'Gradient Boosting':   'GradientBoostingClassifier.pkl',
}

# Actual accuracies from your latest trained models
MODEL_ACCURACIES = {
    'Decision Trees':      84.78,
    'Logistic Regression': 84.24,
    'Random Forest':       89.13,
    'SVM':                 88.04,
    'XGBoost':             86.41,
    'Gradient Boosting':   87.50,
}

def load_pkl(path):
    # First try relative to this file's directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_dir, path)
    if os.path.exists(full_path):
        with open(full_path, 'rb') as f:
            return pickle.load(f)
    # Fallback to current working directory checking
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

def clinical_risk_score(d):
    """
    Evidence-based clinical risk scoring aligned with heart disease literature.
    Feature weights derived from correlation analysis in your ip-2.py:
    ST_Slope > ExerciseAngina > Oldpeak > ChestPainType > MaxHR > Age > Sex
    """
    score = 0.0
    # ST_Slope — highest correlation with HeartDisease in your dataset
    sl = int(d.get('stslope', 0))
    if sl == 2:   score += 3.0   # Downsloping — strongest risk marker
    elif sl == 1: score += 1.9   # Flat
    # Exercise-Induced Angina — strong independent predictor
    if int(d.get('angina', 0)) == 1: score += 2.5
    # Oldpeak / ST Depression
    op = float(d.get('oldpeak', 0))
    if op >= 3:   score += 2.4
    elif op >= 2: score += 1.7
    elif op >= 1: score += 1.0
    elif op > 0:  score += 0.3
    # Chest Pain Type — Asymptomatic = most dangerous (paradox)
    cp = int(d.get('chestpain', 0))
    if cp == 2:   score += 2.6   # Asymptomatic
    elif cp == 1: score += 0.5   # Non-Anginal
    elif cp == 3: score += 0.2   # Typical Angina (least risky paradoxically)
    # Max Heart Rate — inverse relationship
    hr = int(d.get('maxhr', 150))
    if hr < 100:   score += 2.0
    elif hr < 120: score += 1.4
    elif hr < 140: score += 0.7
    elif hr < 155: score += 0.2
    # Age
    age = int(d.get('age', 50))
    if age > 65:   score += 1.6
    elif age > 58: score += 1.1
    elif age > 50: score += 0.6
    elif age > 42: score += 0.2
    # Sex (male statistically higher risk after 45)
    if int(d.get('sex', 0)) == 0 and age > 42: score += 0.6
    # Resting ECG
    ecg = int(d.get('ecg', 0))
    if ecg == 2:   score += 1.0  # LV Hypertrophy
    elif ecg == 1: score += 0.7  # ST-T Abnormality
    # Fasting Blood Sugar > 120
    if int(d.get('fbs', 0)) == 1: score += 0.7
    # Cholesterol
    chol = int(d.get('chol', 200))
    if chol > 320:   score += 1.0
    elif chol > 270: score += 0.6
    elif chol > 240: score += 0.3
    # Resting BP
    rbp = int(d.get('rbp', 120))
    if rbp > 170:   score += 0.8
    elif rbp > 150: score += 0.5
    elif rbp > 140: score += 0.2
    # Max possible score ~18; normalise to 0-1
    return min(score / 18.0, 1.0)

# ─────────────────────────────────────────────────────────────────────
# Routes — Auth
# ─────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return redirect(url_for('dashboard') if 'user_id' in session else url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        email    = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        user = User.query.filter_by(email=email).first()
        if not user or not check_password_hash(user.password_hash, password):
            error = 'Invalid email or password. Please try again.'
        else:
            session.permanent = True
            session['user_id'] = user.id
            session['name']    = user.name
            session['email']   = user.email
            return redirect(url_for('dashboard'))
    return render_template('login.html', error=error)

@app.route('/register', methods=['GET', 'POST'])
def register():
    error   = None
    success = None
    if request.method == 'POST':
        step  = request.form.get('step', 'send_otp')
        email = request.form.get('email', '').strip().lower()

        if step == 'send_otp':
            name     = request.form.get('name', '').strip()
            password = request.form.get('password', '')
            confirm  = request.form.get('confirm', '')
            if not all([name, email, password, confirm]):
                error = 'All fields are required.'
            elif User.query.filter_by(email=email).first():
                error = 'An account with this email already exists.'
            elif password != confirm:
                error = 'Passwords do not match.'
            elif len(password) < 6:
                error = 'Password must be at least 6 characters long.'
            else:
                code = gen_otp()
                OTP.query.filter_by(email=email, used=False).delete()
                db.session.add(OTP(
                    email=email, code=code,
                    expires_at=datetime.utcnow() + timedelta(minutes=10)
                ))
                db.session.commit()
                sent, err_msg = send_otp_via_emailjs(email, name, code)
                # Pass dev_otp only when EmailJS is not yet configured
                dev_otp = code if (err_msg == 'emailjs_not_configured') else None
                return render_template('register.html', step='verify',
                                       email=email, name=name, password=password,
                                       sent=sent, dev_otp=dev_otp)

        elif step == 'verify_otp':
            name     = request.form.get('name', '').strip()
            password = request.form.get('password', '')
            entered  = request.form.get('otp', '').strip()
            record   = OTP.query.filter_by(email=email, used=False)\
                                .order_by(OTP.id.desc()).first()
            if not record or record.code != entered:
                error = 'Invalid OTP code. Please check and try again.'
                return render_template('register.html', step='verify',
                                       email=email, name=name,
                                       password=password, error=error)
            if record.expires_at < datetime.utcnow():
                error = 'OTP has expired. Please go back and request a new one.'
                return render_template('register.html', step='verify',
                                       email=email, name=name,
                                       password=password, error=error)
            record.used = True
            db.session.add(User(
                name=name, email=email,
                password_hash=generate_password_hash(password)
            ))
            db.session.commit()
            success = f'Account created successfully! You can now sign in, {name}.'

    return render_template('register.html', step='email',
                           error=error, success=success)

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    error = None
    if request.method == 'POST':
        step  = request.form.get('step', 'send_otp')
        email = request.form.get('email', '').strip().lower()

        if step == 'send_otp':
            user = User.query.filter_by(email=email).first()
            if not user:
                error = 'No account found with that email address.'
            else:
                code = gen_otp()
                OTP.query.filter_by(email=email, used=False).delete()
                db.session.add(OTP(
                    email=email, code=code,
                    expires_at=datetime.utcnow() + timedelta(minutes=10)
                ))
                db.session.commit()
                sent, err_msg = send_otp_via_emailjs(email, user.name, code)
                dev_otp = code if (err_msg == 'emailjs_not_configured') else None
                return render_template('forgot_password.html', step='verify',
                                       email=email, sent=sent, dev_otp=dev_otp)

        elif step == 'verify_otp':
            password = request.form.get('password', '')
            confirm  = request.form.get('confirm', '')
            entered  = request.form.get('otp', '').strip()
            
            if password != confirm:
                error = 'Passwords do not match.'
            elif len(password) < 6:
                error = 'Password must be at least 6 characters long.'
            else:
                record = OTP.query.filter_by(email=email, used=False)\
                                    .order_by(OTP.id.desc()).first()
                if not record or record.code != entered:
                    error = 'Invalid OTP code. Please try again.'
                elif record.expires_at < datetime.utcnow():
                    error = 'OTP has expired. Please go back and request a new one.'
                else:
                    record.used = True
                    user = User.query.filter_by(email=email).first()
                    user.password_hash = generate_password_hash(password)
                    db.session.commit()
                    return render_template('login.html', error=None, 
                        success="Password has been reset successfully! You can now sign in.")
                
                # if there was an error in verify
                return render_template('forgot_password.html', step='verify',
                                       email=email, error=error)
                
    return render_template('forgot_password.html', step='email', error=error)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# ─────────────────────────────────────────────────────────────────────
# Routes — Main Pages
# ─────────────────────────────────────────────────────────────────────

@app.route('/dashboard')
@login_required
def dashboard():
    user        = db.session.get(User, session['user_id'])
    total_preds = len(user.predictions)
    high_risk   = sum(1 for p in user.predictions if p.risk_label == 'high')
    low_risk    = sum(1 for p in user.predictions if p.risk_label == 'low')
    return render_template('dashboard.html', name=session['name'],
                           total_preds=total_preds,
                           high_risk=high_risk, low_risk=low_risk)

@app.route('/predict')
@login_required
def predict():
    return render_template('predict.html', name=session['name'])

@app.route('/history')
@login_required
def history():
    preds = Prediction.query.filter_by(user_id=session['user_id'])\
                            .order_by(Prediction.created_at.desc()).all()
    return render_template('history.html', predictions=preds, name=session['name'])

@app.route('/reports')
@login_required
def reports():
    preds = Prediction.query.filter_by(user_id=session['user_id'])\
                            .order_by(Prediction.created_at.desc()).all()
    return render_template('reports.html', predictions=preds, name=session['name'])

@app.route('/performance')
@login_required
def performance():
    return render_template('performance.html', name=session['name'])

# ─────────────────────────────────────────────────────────────────────
# Routes — Prediction API
# ─────────────────────────────────────────────────────────────────────
@app.route('/api/predict', methods=['POST'])
@login_required
def api_predict():
    data = request.get_json()
    try:
        input_df = pd.DataFrame([{
            'Age':            int(data['age']),
            'Sex':            int(data['sex']),
            'ChestPainType':  int(data['chestpain']),
            'RestingBP':      int(data['rbp']),
            'Cholesterol':    int(data['chol']),
            'FastingBS':      int(data['fbs']),
            'RestingECG':     int(data['ecg']),
            'MaxHR':          int(data['maxhr']),
            'ExerciseAngina': int(data['angina']),
            'Oldpeak':        float(data['oldpeak']),
            'ST_Slope':       int(data['stslope']),
        }])

        # Add new engineered features to match training script (ip-2-updated.py)
        input_df['AgeMaxHR_ratio']     = input_df['Age'] / (input_df['MaxHR'] + 1)
        input_df['OldpeakSlope']       = input_df['Oldpeak'] * input_df['ST_Slope']
        input_df['AnginaChestPain']    = input_df['ExerciseAngina'] * input_df['ChestPainType']
        input_df['HighBP_flag']        = (input_df['RestingBP'] > 140).astype(int)
        input_df['HighChol_flag']      = (input_df['Cholesterol'] > 240).astype(int)
        input_df['LowHR_flag']         = (input_df['MaxHR'] < 130).astype(int)
        input_df['Age_Risk']           = ((input_df['Age'] > 55) & (input_df['Sex'] == 0)).astype(int)

        scaler = load_pkl('scaler.pkl')
        if scaler:
            input_scaled = scaler.transform(input_df)
        else:
            input_scaled = input_df

    except Exception as e:
        return jsonify({'error': f'Input error: {str(e)}'}), 400

    base_risk = clinical_risk_score(data)
    results   = []
    any_demo  = False

    for model_name, fname in ML_MODELS.items():
        model = load_pkl(fname)
        if model:
            # Real model prediction
            try:
                pred = int(model.predict(input_scaled)[0])
                prob = None
                if hasattr(model, 'predict_proba'):
                    prob = round(float(model.predict_proba(input_scaled)[0][1]) * 100, 1)
                results.append({'model': model_name, 'prediction': pred,
                                'probability': prob, 'demo': False})
            except Exception as e:
                results.append({'model': model_name, 'prediction': -1,
                                'error': str(e), 'demo': False})
        else:
            # Demo mode — simulate using clinical risk score
            # Different models have different sensitivity/specificity trade-offs
            any_demo = True
            acc = MODEL_ACCURACIES.get(model_name, 85.0) / 100.0
            # Add small per-model variance so results look realistic
            model_bias = {
                'Decision Trees':      0.02,
                'Logistic Regression': -0.01,
                'Random Forest':       0.0,
                'SVM':                 0.01,
                'XGBoost':             -0.02,
                'Gradient Boosting':   0.01,
            }
            bias  = model_bias.get(model_name, 0)
            noise = random.gauss(0, 0.04)  # small Gaussian noise
            prob  = round(max(2, min(98, (base_risk + bias + noise) * 100)), 1)
            pred  = 1 if prob >= 50 else 0
            results.append({'model': model_name, 'prediction': pred,
                            'probability': prob, 'demo': True})

    valid      = [r for r in results if r.get('prediction', -1) != -1]
    risk_count = sum(1 for r in valid if r['prediction'] == 1)
    risk_pct   = round(risk_count / len(valid) * 100) if valid else 0
    risk_label = 'high' if risk_pct > 60 else ('moderate' if risk_pct > 35 else 'low')

    pred_record = Prediction(
        user_id=session['user_id'],
        age=int(data['age']),         sex=int(data['sex']),
        chestpain=int(data['chestpain']), rbp=int(data['rbp']),
        cholesterol=int(data['chol']),    fasting_bs=int(data['fbs']),
        ecg=int(data['ecg']),             max_hr=int(data['maxhr']),
        angina=int(data['angina']),       oldpeak=float(data['oldpeak']),
        st_slope=int(data['stslope']),    risk_pct=risk_pct,
        risk_label=risk_label,            results_json=json.dumps(results),
    )
    db.session.add(pred_record)
    db.session.commit()

    return jsonify({'results': results, 'risk_pct': risk_pct,
                    'risk_count': risk_count, 'total': len(valid),
                    'risk_label': risk_label, 'pred_id': pred_record.id,
                    'demo': any_demo})

# ─────────────────────────────────────────────────────────────────────
# Routes — PDF Report
# ─────────────────────────────────────────────────────────────────────
CP_MAP  = {0:'Atypical Angina', 1:'Non-Anginal Pain', 2:'Asymptomatic', 3:'Typical Angina'}
ECG_MAP = {0:'Normal', 1:'ST-T Wave Abnormality', 2:'Left Ventricular Hypertrophy'}
SL_MAP  = {0:'Upsloping', 1:'Flat', 2:'Downsloping'}
SEX_MAP = {0:'Male', 1:'Female'}
YN_MAP  = {0:'No', 1:'Yes'}
FBS_MAP = {0:'≤ 120 mg/dl', 1:'> 120 mg/dl'}

@app.route('/api/report/<int:pred_id>')
@login_required
def download_report(pred_id):
    pred = db.session.get(Prediction, pred_id)
    if not pred:      abort(404)
    if pred.user_id != session['user_id']: abort(403)
    user = db.session.get(User, session['user_id'])

    buf  = io.BytesIO()
    doc  = SimpleDocTemplate(buf, pagesize=A4,
                             leftMargin=2*cm, rightMargin=2*cm,
                             topMargin=2*cm, bottomMargin=2*cm)
    RED   = colors.HexColor('#e03050')
    DARK  = colors.HexColor('#0f0f14')
    LGREY = colors.HexColor('#f5f5f7')

    def ps(nm, **kw):
        base = getSampleStyleSheet()['Normal']
        return ParagraphStyle(nm, parent=base, **kw)

    story = [
        Paragraph('♥ CardioSense', ps('T', fontName='Helvetica-Bold',
                  fontSize=24, textColor=RED, alignment=TA_CENTER, spaceAfter=2)),
        Paragraph('Cardiovascular Risk Assessment Report', ps('S', fontName='Helvetica',
                  fontSize=10, textColor=colors.grey, alignment=TA_CENTER, spaceAfter=4)),
        Spacer(1, 0.2*cm),
        HRFlowable(width='100%', thickness=2, color=RED),
        Spacer(1, 0.3*cm),
    ]

    # Patient info
    story.append(Paragraph('Patient Information',
        ps('H2', fontName='Helvetica-Bold', fontSize=13, textColor=DARK,
           spaceBefore=10, spaceAfter=5)))
    info = Table([
        ['Patient Name', user.name,  'Report ID',  f'#{pred.id:05d}'],
        ['Email',  user.email,       'Date',  pred.created_at.strftime('%d %b %Y, %H:%M')],
        ['Member Since', user.created_at.strftime('%d %b %Y'),
         'Total Predictions', str(len(user.predictions))],
    ], colWidths=[3.5*cm, 6.5*cm, 3.2*cm, 4.8*cm])
    info.setStyle(TableStyle([
        ('FONTNAME', (0,0),(-1,-1),'Helvetica'),
        ('FONTNAME', (0,0),(0,-1),'Helvetica-Bold'),
        ('FONTNAME', (2,0),(2,-1),'Helvetica-Bold'),
        ('FONTSIZE', (0,0),(-1,-1), 9),
        ('TEXTCOLOR',(0,0),(0,-1), colors.grey),
        ('TEXTCOLOR',(2,0),(2,-1), colors.grey),
        ('ROWBACKGROUNDS',(0,0),(-1,-1),[LGREY, colors.white]),
        ('GRID',(0,0),(-1,-1),0.3,colors.HexColor('#ddd')),
        ('PADDING',(0,0),(-1,-1),6),
    ]))
    story += [info, Spacer(1, 0.4*cm)]

    # Risk banner
    risk_col = (colors.HexColor('#e03050') if pred.risk_label == 'high' else
                colors.HexColor('#d97706') if pred.risk_label == 'moderate' else
                colors.HexColor('#16a34a'))
    rl_txt = {'high':'⚠  HIGH RISK','moderate':'⚡  MODERATE RISK','low':'✓  LOW RISK'}
    story.append(Paragraph('Risk Assessment Result',
        ps('H2b', fontName='Helvetica-Bold', fontSize=13, textColor=DARK,
           spaceBefore=10, spaceAfter=5)))
    res_list = json.loads(pred.results_json)
    risk_models = sum(1 for r in res_list if r.get('prediction') == 1)
    banner = Table([[f'{pred.risk_pct}% Risk Score',
                     rl_txt[pred.risk_label],
                     f'{risk_models} / {len(res_list)} models flagged risk']],
                   colWidths=[5.5*cm, 5.5*cm, 7*cm])
    banner.setStyle(TableStyle([
        ('FONTNAME',(0,0),(-1,-1),'Helvetica-Bold'),
        ('FONTSIZE',(0,0),(-1,-1),11),
        ('TEXTCOLOR',(0,0),(-1,-1),colors.white),
        ('BACKGROUND',(0,0),(-1,-1),risk_col),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('PADDING',(0,0),(-1,-1),12),
    ]))
    story += [banner, Spacer(1, 0.4*cm)]

    # Clinical parameters
    story.append(Paragraph('Clinical Parameters',
        ps('H2c', fontName='Helvetica-Bold', fontSize=13, textColor=DARK,
           spaceBefore=10, spaceAfter=5)))
    params = Table([
        ['Parameter','Value','Parameter','Value'],
        ['Age',            f'{pred.age} years',        'Sex',                SEX_MAP.get(pred.sex,'—')],
        ['Chest Pain Type', CP_MAP.get(pred.chestpain,'—'), 'Resting BP',   f'{pred.rbp} mm Hg'],
        ['Cholesterol',    f'{pred.cholesterol} mg/dl', 'Fasting Blood Sugar', FBS_MAP.get(pred.fasting_bs,'—')],
        ['Resting ECG',     ECG_MAP.get(pred.ecg,'—'),  'Max Heart Rate',   f'{pred.max_hr} bpm'],
        ['Exercise Angina', YN_MAP.get(pred.angina,'—'),'ST Depression',    str(pred.oldpeak)],
        ['ST Slope',        SL_MAP.get(pred.st_slope,'—'), '', ''],
    ], colWidths=[4.5*cm,5*cm,4.5*cm,4*cm])
    params.setStyle(TableStyle([
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
        ('FONTNAME',(0,1),(0,-1),'Helvetica-Bold'),
        ('FONTNAME',(2,1),(2,-1),'Helvetica-Bold'),
        ('FONTSIZE',(0,0),(-1,-1),9),
        ('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('TEXTCOLOR',(0,1),(0,-1),colors.grey),
        ('TEXTCOLOR',(2,1),(2,-1),colors.grey),
        ('BACKGROUND',(0,0),(-1,0),DARK),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[LGREY,colors.white]),
        ('GRID',(0,0),(-1,-1),0.3,colors.HexColor('#ddd')),
        ('PADDING',(0,0),(-1,-1),7),
    ]))
    story += [params, Spacer(1, 0.4*cm)]

    # Model results
    story.append(Paragraph('Individual Model Results',
        ps('H2d', fontName='Helvetica-Bold', fontSize=13, textColor=DARK,
           spaceBefore=10, spaceAfter=5)))
    m_rows = [['Model','Prediction','Probability','Verdict']]
    for r in res_list:
        ir = r.get('prediction') == 1
        m_rows.append([r['model'],
                        'Heart Disease' if ir else 'No Disease',
                        f"{r['probability']}%" if r.get('probability') is not None else '—',
                        '⚠ Risk' if ir else '✓ Safe'])
    m_tbl = Table(m_rows, colWidths=[5.5*cm,4*cm,3.5*cm,5*cm])
    row_styles = [
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
        ('FONTSIZE',(0,0),(-1,-1),9),
        ('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('BACKGROUND',(0,0),(-1,0),DARK),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[LGREY,colors.white]),
        ('GRID',(0,0),(-1,-1),0.3,colors.HexColor('#ddd')),
        ('PADDING',(0,0),(-1,-1),7),
    ]
    for i, r in enumerate(res_list, 1):
        ir = r.get('prediction') == 1
        row_styles += [
            ('TEXTCOLOR',(3,i),(3,i), colors.HexColor('#e03050') if ir else colors.HexColor('#16a34a')),
            ('FONTNAME',(3,i),(3,i),'Helvetica-Bold'),
        ]
    m_tbl.setStyle(TableStyle(row_styles))
    story += [m_tbl, Spacer(1, 0.4*cm)]

    # Recommendations
    recs = {
        'high': [
            '• Schedule an urgent appointment with a cardiologist for a full cardiac evaluation.',
            '• Undergo ECG, echocardiogram, and stress testing as soon as possible.',
            '• Monitor blood pressure and cholesterol levels daily.',
            '• Avoid strenuous physical activity until you have been medically assessed.',
            '• Begin a heart-healthy diet: reduce sodium, saturated fat, and processed foods.',
            '• If you smoke, stop immediately — smoking is a major modifiable risk factor.',
        ],
        'moderate': [
            '• Schedule a cardiac checkup with your primary physician within 2–4 weeks.',
            '• Adopt a heart-healthy diet rich in fruits, vegetables, and whole grains.',
            '• Aim for at least 150 minutes of moderate aerobic exercise each week.',
            '• Check blood pressure and cholesterol every 3 months.',
            '• Manage stress through mindfulness, yoga, or regular relaxation.',
        ],
        'low': [
            '• Continue your current healthy lifestyle — you are doing well.',
            '• Schedule an annual health screening as a preventive measure.',
            '• Maintain a balanced diet and regular exercise routine.',
            '• Report any new cardiac symptoms (chest pain, breathlessness) to a doctor promptly.',
            '• Stay well hydrated and ensure sufficient sleep each night.',
        ],
    }
    story.append(Paragraph('Clinical Recommendations',
        ps('H2e', fontName='Helvetica-Bold', fontSize=13, textColor=DARK,
           spaceBefore=10, spaceAfter=5)))
    body_s = ps('Bd', fontName='Helvetica', fontSize=9.5,
                textColor=colors.black, spaceAfter=4, leading=15)
    for rec in recs[pred.risk_label]:
        story.append(Paragraph(rec, body_s))

    story += [
        Spacer(1, 0.4*cm),
        HRFlowable(width='100%', thickness=1, color=colors.HexColor('#ddd')),
        Spacer(1, 0.25*cm),
        Paragraph(
            '⚠ Medical Disclaimer: This report is generated by an AI-powered tool for '
            'educational and clinical support purposes only. It does not constitute a medical '
            'diagnosis or treatment recommendation. Always consult a qualified healthcare professional.',
            ps('Dc', fontName='Helvetica', fontSize=7.5, textColor=colors.grey,
               alignment=TA_CENTER, leading=11)
        ),
    ]

    doc.build(story)
    buf.seek(0)
    fname = f"CardioSense_Report_{pred.id:05d}_{pred.created_at.strftime('%Y%m%d')}.pdf"
    return send_file(buf, as_attachment=True, download_name=fname,
                     mimetype='application/pdf')

if __name__ == '__main__':
    app.run(debug=True)


















