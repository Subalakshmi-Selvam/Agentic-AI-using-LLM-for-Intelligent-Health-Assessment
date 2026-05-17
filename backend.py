from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import bcrypt
import os
from werkzeug.utils import secure_filename

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
model1 = load_model('skin_disease_model.h5')
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# ---------- helper ----------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------- DB CONNECTION ----------
def get_db():
    conn = sqlite3.connect("mydb.db")
    conn.row_factory = sqlite3.Row  # enables dict-like access
    return conn
@app.route("/api/register", methods=["POST"])
def register():
    data = request.json
    name = data.get("name")
    email = data.get("email")
    password = data.get("password")

    if not name or not email or not password:
        return jsonify({"message": "All fields are required"}), 400

    db = get_db()
    cursor = db.cursor()

    cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
    if cursor.fetchone():
        return jsonify({"message": "Email already exists"}), 400

    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())

    cursor.execute(
        "INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
        (name, email, hashed.decode("utf-8"))
    )
    db.commit()

    return jsonify({"message": "Account created successfully"})
@app.route("/api/login", methods=["POST"])
def login():
    data = request.json
    email = data.get("email")
    password = data.get("password")

    if not email or not password:
        return jsonify({"message": "Email and password required"}), 400

    db = get_db()
    cursor = db.cursor()

    cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
    user = cursor.fetchone()

    if not user:
        return jsonify({"message": "Invalid credentials"}), 401

    stored_password = user["password"].encode("utf-8")

    if not bcrypt.checkpw(password.encode("utf-8"), stored_password):
        return jsonify({"message": "Invalid credentials"}), 401

    return jsonify({
        "message": "Login successful",
        "user": {
            "id": user["id"],
            "name": user["name"],
            "email": user["email"]
        }
    })
@app.route("/api/profile/<int:user_id>", methods=["GET"])
def get_profile(user_id):
    db = get_db()
    cursor = db.cursor()

    cursor.execute("SELECT id, name, email FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()

    if not user:
        return jsonify({"message": "User not found"}), 404

    return jsonify({
        "id": user["id"],
        "name": user["name"],
        "email": user["email"]
    })

@app.route("/api/profile/update", methods=["PUT"])
def update_profile():
    data = request.json
    user_id = data.get("id")
    name = data.get("name")
    email = data.get("email")

    if not user_id or not name or not email:
        return jsonify({"message": "Missing fields"}), 400

    db = get_db()
    cursor = db.cursor()

    cursor.execute(
        "UPDATE users SET name = ?, email = ? WHERE id = ?",
        (name, email, user_id)
    )
    db.commit()

    return jsonify({"message": "Profile updated successfully"})


def predict(image_path):    
    
    # Define the path to the image
    # image_path=r'C:\Users\Dell\OneDrive\Desktop\skin\archive (4)\Skin cancer ISIC The International Skin Imaging Collaboration\Test\squamous cell carcinoma\ISIC_0011593.jpg'

    # Load the image and preprocess it
    img = image.load_img(image_path, target_size=(75, 100))  # Correct order: height=75px, width=100px
    img_array = image.img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Normalize the image as per the training data preprocessing
    img_array = (img_array - np.mean(img_array)) / np.std(img_array)
    # Predict the class
    predictions = model1.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    label_map={0: 'actinic keratosis',
    1: 'basal cell carcinoma',
    2: 'dermatofibroma',
    3: 'melanoma',
    4: 'nevus',
    5: 'pigmented benign keratosis',
    6: 'seborrheic keratosis',
    7: 'squamous cell carcinoma',
    8: 'vascular lesion'}
    # Map the predicted class index to the class label
    predicted_label = label_map[predicted_class[0]]
    print(f"Predicted Class: {predicted_label}")
    return predicted_label

@app.route("/api/predict", methods=["POST"])
def predict_api():
    if 'file' not in request.files:
        return jsonify({"message": "No file uploaded"}), 400

    file = request.files['file']

    if file.filename == "":
        return jsonify({"message": "No selected file"}), 400
    SUGGESTIONS = {
    "actinic keratosis": {
        "description": "Actinic keratosis is a rough, scaly patch on the skin caused by long-term sun exposure.",
        "symptoms": "Dry, scaly, or crusty patches that may itch or burn.",
        "causes": "Prolonged exposure to ultraviolet (UV) radiation from sunlight.",
        "treatments": "Cryotherapy, topical creams, photodynamic therapy.",
        "action": "Consult a dermatologist for early treatment."
    },

    "basal cell carcinoma": {
        "description": "Basal cell carcinoma is a common type of skin cancer that grows slowly and rarely spreads.",
        "symptoms": "Pearly or waxy bumps, flat flesh-colored lesions.",
        "causes": "Long-term UV exposure and genetic factors.",
        "treatments": "Surgical removal, Mohs surgery, radiation therapy.",
        "action": "Seek medical attention immediately."
    },

    "dermatofibroma": {
        "description": "Dermatofibroma is a benign skin growth often found on the legs or arms.",
        "symptoms": "Firm, raised nodules that may be brown or reddish.",
        "causes": "Minor skin injuries such as insect bites or trauma.",
        "treatments": "Usually none required; surgical removal if painful.",
        "action": "Consult dermatologist if growth changes."
    },

    "melanoma": {
        "description": "Melanoma is a serious form of skin cancer that develops in pigment-producing cells.",
        "symptoms": "Irregular moles with uneven color, shape, or size.",
        "causes": "Excessive UV exposure and genetic predisposition.",
        "treatments": "Surgery, immunotherapy, targeted therapy.",
        "action": "Urgent dermatological consultation required."
    },

    "nevus": {
        "description": "Nevus is a common benign mole usually harmless.",
        "symptoms": "Small brown or black spots on the skin.",
        "causes": "Clusters of melanocytes; genetic factors.",
        "treatments": "No treatment needed unless cosmetic or changes occur.",
        "action": "Monitor for changes regularly."
    },

    "pigmented benign keratosis": {
        "description": "A non-cancerous pigmented skin lesion common in older adults.",
        "symptoms": "Dark, raised, wart-like growths.",
        "causes": "Aging and genetic predisposition.",
        "treatments": "Cryotherapy or laser removal if needed.",
        "action": "Consult dermatologist for confirmation."
    },

    "seborrheic keratosis": {
        "description": "Seborrheic keratosis is a harmless skin growth with a waxy appearance.",
        "symptoms": "Brown, black, or tan raised patches.",
        "causes": "Age-related skin changes.",
        "treatments": "No treatment required unless irritated.",
        "action": "Dermatologist visit if sudden growth occurs."
    },

    "squamous cell carcinoma": {
        "description": "A common form of skin cancer that may spread if untreated.",
        "symptoms": "Scaly red patches, open sores, or thickened skin.",
        "causes": "UV radiation exposure, weakened immune system.",
        "treatments": "Surgery, radiation therapy, topical medications.",
        "action": "Immediate medical evaluation recommended."
    },

    "vascular lesion": {
        "description": "Vascular lesions are abnormalities of blood vessels in the skin.",
        "symptoms": "Red, purple, or blue marks on the skin.",
        "causes": "Congenital conditions or trauma.",
        "treatments": "Laser therapy or medical monitoring.",
        "action": "Consult dermatologist for assessment."
    }
}

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)

        label = predict(path)

        return jsonify({
            "prediction": label.upper(),
            "image": filename,
            "details": SUGGESTIONS[label]
        })

    return jsonify({"message": "Invalid file type"}), 400
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5005)