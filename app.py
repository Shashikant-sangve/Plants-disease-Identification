from flask import Flask, render_template, request, redirect, url_for, session, flash
import os
import sqlite3
from werkzeug.utils import secure_filename
from utils.predict import predict_disease_and_severity  # your model
from utils.gradcam import generate_gradcam  # optional
from datetime import timedelta

# NEW: CV + NumPy for basic leaf gating
import cv2
import numpy as np
import time

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
app.permanent_session_lifetime = timedelta(minutes=30)

# -------- Paths / Folders ----------
UPLOAD_FOLDER = 'static/uploads'
DB_DIR = 'database'
DB_PATH = os.path.join(DB_DIR, 'users.db')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# -------- Allowed file types ----------
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
ALLOWED_MIMES = {'image/png', 'image/jpeg', 'image/webp'}

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# -------- Leaf gate (quick heuristic) ----------
def is_leaf_image(filepath: str) -> bool:
    """
    Heuristic filter to accept likely leaf images and reject others:
    - require enough green/yellow pixels (HSV ranges typical for foliage),
    - reject if strong presence of face or skin-like regions.
    Adjust thresholds as needed for your data.
    """
    img = cv2.imread(filepath)
    if img is None:
        return False

    # Resize to speed up analysis
    h, w = img.shape[:2]
    scale = 600 / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

    # Basic blur to reduce noise
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)

    # HSV masks for green & yellow (typical foliage)
    hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
    # Green range (Hue ~35–85)
    green_lo = np.array([35, 40, 30]); green_hi = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, green_lo, green_hi)

    # Yellow range (for yellowing leaves; Hue ~15–35)
    yellow_lo = np.array([15, 40, 30]); yellow_hi = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, yellow_lo, yellow_hi)

    # Ratios
    total_pixels = img.shape[0] * img.shape[1]
    green_ratio = float(np.count_nonzero(green_mask)) / total_pixels
    yellow_ratio = float(np.count_nonzero(yellow_mask)) / total_pixels

    # Face reject (avoid people photos)
    gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
    has_face = len(faces) > 0

    # Skin-tone reject (YCrCb range)
    ycrcb = cv2.cvtColor(img_blur, cv2.COLOR_BGR2YCrCb)
    skin_lo = np.array([0, 133, 77]); skin_hi = np.array([255, 173, 127])
    skin_mask = cv2.inRange(ycrcb, skin_lo, skin_hi)
    skin_ratio = float(np.count_nonzero(skin_mask)) / total_pixels

    # Simple texture/entropy check to avoid flat/blank images
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = float(np.count_nonzero(edges)) / total_pixels

    # ---- Thresholds (tune if needed) ----
    foliage_ratio = green_ratio + (0.7 * yellow_ratio)  # weight yellow a bit less
    foliage_ok = foliage_ratio >= 0.12                  # ~12% foliage pixels
    not_face = not has_face
    skin_ok = skin_ratio <= 0.25                        # reject if >25% skin-like
    texture_ok = edge_ratio >= 0.01                     # avoid blank images

    return foliage_ok and not_face and skin_ok and texture_ok

# ---------------- DB SETUP -------------------
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS users (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            username TEXT UNIQUE,
                            password TEXT
                        );''')

init_db()

# ---------------- ROUTES ---------------------

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']
        if not username or not password:
            flash("Username and password are required.", "danger")
            return redirect(request.url)
        try:
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
                conn.commit()
            flash("Signup successful, please log in.", "success")
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash("Username already exists.", "danger")
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
            user = cur.fetchone()
            if user:
                session.permanent = True
                session['user'] = username
                flash("Login successful.", "success")
                return redirect(url_for('upload'))
            else:
                flash("Invalid credentials.", "danger")
    return render_template('login.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'user' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        file = request.files.get('leaf_image')

        # Basic validations
        if not file or file.filename == '':
            flash("No file selected. Please choose an image.", "danger")
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash("Only image files are allowed (JPG, JPEG, PNG, WEBP).", "danger")
            return redirect(request.url)

        if file.mimetype not in ALLOWED_MIMES:
            flash("Invalid image type. Please upload a valid image.", "danger")
            return redirect(request.url)

        # Unique filename to avoid clashes
        name, ext = os.path.splitext(secure_filename(file.filename))
        filename = f"{name}_{int(time.time())}{ext.lower()}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # ---- LEAF GATE ----
        if not is_leaf_image(filepath):
            # Clean up and reject
            try:
                os.remove(filepath)
            except Exception:
                pass
            flash("Only leaf photos are allowed. Please upload a clear leaf image (avoid people/skin/background-only).", "danger")
            return redirect(request.url)

        # If it passes the gate, run your model
        result, severity, gradcam_filename, confidence, top_classes, top_confidences, info_dict = \
            predict_disease_and_severity(filepath)

        model_metrics = {
            'Accuracy': 0.92,
            'Precision': 0.91,
            'Recall': 0.90,
            'F1 Score': 0.905
        }

        return render_template(
            'result.html',
            filename=filename,
            result=result,
            severity=severity,
            confidence=confidence,
            top_classes=top_classes,
            top_confidences=top_confidences,
            disease_info=info_dict,
            gradcam_img=gradcam_filename,
            model_metrics=model_metrics
        )

    return render_template('upload.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    flash("Logged out successfully.", "info")
    return redirect(url_for('login'))

# ---------------- MAIN ---------------------
if __name__ == '__main__':
    app.run(debug=True)