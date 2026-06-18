import os
import io
import re
import uuid
import random
import sqlite3
import threading
from datetime import datetime

import gc
import numpy as np

from flask import (Flask, render_template, request, redirect,
                   url_for, session, flash, jsonify, send_file)
from werkzeug.utils import secure_filename

# LangChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# ReportLab
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, Image)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

import google.generativeai as genai

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
genai.configure(api_key=GOOGLE_API_KEY)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "secret key")

UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

INDEX_FOLDER = "faiss_index"
os.makedirs(INDEX_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ─────────────────────────────────────────
# LAZY RAG GLOBALS
# ─────────────────────────────────────────
_rag_retriever = None
_rag_llm = None
_rag_prompt = None
_rag_lock = threading.Lock()


def get_embeddings():
    """Google embeddings — no PyTorch, no sentence-transformers, ~0 extra RAM."""
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )


def _init_rag():
    global _rag_retriever, _rag_llm, _rag_prompt
    with _rag_lock:
        if _rag_retriever is not None:
            return
        if not os.path.exists("report.pdf"):
            print("WARNING: report.pdf not found – RAG features disabled.")
            return
        import time
        print("Loading Cancer PDF …")
        loader = PyPDFLoader("report.pdf")
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
        docs = splitter.split_documents(documents)
        print(f"Embedding {len(docs)} chunks in batches …")
        embeddings = get_embeddings()
        # Embed in small batches to avoid Google API 504 timeout
        BATCH = 20
        vectorstore = None
        for i in range(0, len(docs), BATCH):
            batch = docs[i:i + BATCH]
            if vectorstore is None:
                vectorstore = FAISS.from_documents(batch, embeddings)
            else:
                vectorstore.add_documents(batch)
            print(f"  embedded {min(i + BATCH, len(docs))}/{len(docs)}")
            time.sleep(1)
        _rag_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        _rag_llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0,
            google_api_key=GOOGLE_API_KEY
        )
        _rag_prompt = ChatPromptTemplate.from_template("""
You are an expert oncology medical AI.

STRICT RULES:
- Use ONLY the provided context.
- If answer not found, say: "Information not available in the document."
- Do not hallucinate.

Context:
{context}

Question:
{question}

Provide structured output:
1. Explanation:
2. Symptoms:
3. Risk Factors:
4. Prevention:
5. Additional Notes:
""")
        print("RAG ready.")
        gc.collect()  # free memory used during PDF loading


# Preload RAG in background so first request is fast
threading.Thread(target=_init_rag, daemon=True).start()

# ─────────────────────────────────────────
# LAZY SKIN MODEL
# ─────────────────────────────────────────
_skin_model = None
_skin_lock = threading.Lock()


def _get_skin_model():
    global _skin_model
    with _skin_lock:
        if _skin_model is None:
            if os.path.exists('skin_disease_model.h5'):
                # tf_keras loads legacy Keras 2 .h5 models (avoids layer name slash error)
                import tf_keras
                _skin_model = tf_keras.models.load_model('skin_disease_model.h5')
            else:
                print("WARNING: skin_disease_model.h5 not found.")
    return _skin_model


# ─────────────────────────────────────────
# PDF-UPLOAD CHAT STORE
# ─────────────────────────────────────────
vectorstore1 = None


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def create_pdf_from_text(text, output_path):
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter
    y = height - 40
    for line in text.split("\n"):
        c.drawString(40, y, line[:100])
        y -= 20
        if y < 40:
            c.showPage()
            y = height - 40
    c.save()


def extract_text_from_image(image_path):
    model = genai.GenerativeModel("gemini-2.0-flash")
    uploaded_file = genai.upload_file(image_path)
    response = model.generate_content([
        "Extract all text from this image accurately.",
        uploaded_file
    ])
    return response.text


def load_or_create_store(docs):
    embeddings = get_embeddings()
    if os.path.exists(os.path.join(INDEX_FOLDER, "index.faiss")):
        vs = FAISS.load_local(INDEX_FOLDER, embeddings,
                              allow_dangerous_deserialization=True)
        vs.add_documents(docs)
    else:
        vs = FAISS.from_documents(docs, embeddings)
    vs.save_local(INDEX_FOLDER)
    return vs


def load_store_if_exists():
    embeddings = get_embeddings()
    if os.path.exists(os.path.join(INDEX_FOLDER, "index.faiss")):
        return FAISS.load_local(INDEX_FOLDER, embeddings,
                                allow_dangerous_deserialization=True)
    return None


def predict(image_path):
    import tf_keras
    tf_image = tf_keras.preprocessing.image
    model1 = _get_skin_model()
    if model1 is None:
        return "Model not available"
    img = tf_image.load_img(image_path, target_size=(75, 100))
    img_array = tf_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = (img_array - np.mean(img_array)) / np.std(img_array)
    predictions = model1.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    del img_array, predictions  # free memory immediately
    gc.collect()
    label_map = {
        0: 'actinic keratosis', 1: 'basal cell carcinoma',
        2: 'dermatofibroma', 3: 'melanoma', 4: 'nevus',
        5: 'pigmented benign keratosis', 6: 'seborrheic keratosis',
        7: 'squamous cell carcinoma', 8: 'vascular lesion'
    }
    return label_map[predicted_class[0]]


def _build_report_pdf(label, filename, answer, name, age, gender, email):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        name="Title", fontSize=20, alignment=TA_CENTER,
        textColor=colors.darkblue, spaceAfter=10)
    section_style = ParagraphStyle(
        name="Section", fontSize=13, textColor=colors.black, spaceAfter=6)
    normal = styles["Normal"]
    content = []

    content.append(Paragraph("<b>AI SKIN DIAGNOSIS REPORT</b>", title_style))
    line = Table([[""]], colWidths=[450])
    line.setStyle(TableStyle([('LINEBELOW', (0, 0), (-1, -1), 2, colors.orange)]))
    content.append(line)
    content.append(Spacer(1, 15))

    report_id = "REP" + str(random.randint(1000, 9999))
    today = datetime.now().strftime("%d-%m-%Y")
    content.append(Paragraph("<b>Patient Information</b>", section_style))
    patient_table = Table([
        ["Name:", name], ["Email:", email], ["Age:", str(age)],
        ["Gender:", gender], ["Report ID:", report_id], ["Date:", today]
    ], colWidths=[120, 300])
    patient_table.setStyle(TableStyle([('BOTTOMPADDING', (0, 0), (-1, -1), 6)]))
    content.append(patient_table)
    content.append(Spacer(1, 15))

    image_path = os.path.join("static/uploads", filename)
    if os.path.exists(image_path):
        content.append(Paragraph("<b>Uploaded Image</b>", section_style))
        content.append(Image(image_path, width=160, height=160))
        content.append(Spacer(1, 15))

    answer = answer.replace("**", "")

    def get_section(sec_name):
        match = re.search(rf"{sec_name}:(.*?)(\n\d\.|$)", answer, re.S)
        return match.group(1).strip() if match else "Not available"

    content.append(Paragraph("<b>AI Diagnosis</b>", section_style))
    content.append(Paragraph(f"Disease: {label}", normal))
    content.append(Spacer(1, 10))

    for title in ["Explanation", "Symptoms", "Risk Factors", "Prevention", "Additional Notes"]:
        content.append(Paragraph(f"<b>{title}</b>", section_style))
        content.append(Paragraph(get_section(title), normal))
        content.append(Spacer(1, 10))

    doc.build(content)
    buffer.seek(0)
    return buffer


# ─────────────────────────────────────────
# INIT DB
# ─────────────────────────────────────────
def init_db():
    conn = sqlite3.connect("signup.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT, email TEXT UNIQUE,
            age TEXT, gender TEXT, password TEXT)""")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            userid INTEGER, filename TEXT, label TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit()
    conn.close()

init_db()


# ─────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route('/logon')
def logon():
    return render_template('signup.html')

@app.route('/login')
def login():
    return render_template('signin.html')

@app.route("/signup", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        age = request.form["age"]
        gender = request.form["gender"]
        password = request.form["password"]
        try:
            conn = sqlite3.connect("signup.db")
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (name, email, age, gender, password) VALUES (?, ?, ?, ?, ?)",
                (name, email, age, gender, password))
            conn.commit()
            conn.close()
            flash("Registration Successful! Please login.", "success")
            return redirect(url_for("login"))
        except Exception:
            flash("Email already exists!", "danger")
    return redirect("/login")

@app.route("/signin", methods=["GET", "POST"])
def signin():
    if request.method == "POST":
        username = request.form.get("user")
        password = request.form.get("password")
        if username == "admin" and password == "admin":
            session["admin"] = True
            return redirect(url_for("admin_dashboard"))
        con = sqlite3.connect("signup.db")
        cur = con.cursor()
        cur.execute(
            "SELECT id, name, password FROM users WHERE email=? AND password=?",
            (username, password))
        data = cur.fetchone()
        con.close()
        if data:
            session["id"] = data[0]
            session["user"] = data[1]
            return redirect(url_for("home"))
        else:
            flash("Invalid username or password")
            return render_template("signin.html")
    return render_template("signin.html")

@app.route('/home')
def home():
    return render_template('dashboard.html', user=session["user"])

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/about")
def about():
    return render_template("about.html", user=session["user"])

@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        print(request.form.get("name"), request.form.get("email"), request.form.get("message"))
        return "Message Sent Successfully ✅"
    return render_template("contact.html", user=session["user"])

@app.route("/upload")
def upload():
    return render_template("home.html", user=session["user"])

@app.route("/chatbot")
def chatbot():
    return render_template("chatbot.html", user=session["user"])

@app.route("/chatupload")
def chatupload():
    return render_template("aichat.html", user=session["user"])


# ── Skin prediction ──────────────────────
@app.route('/store', methods=['POST'])
def upload_image():
    if "id" not in session:
        flash("Please login first")
        return redirect(url_for("login"))
    if 'file' not in request.files:
        flash('No file uploaded')
        return redirect(url_for("home"))
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        flash('Invalid file')
        return redirect(url_for("home"))
    filename = str(uuid.uuid4()) + "_" + secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    label = predict(file_path)
    try:
        conn = sqlite3.connect("signup.db")
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO predictions (userid, filename, label) VALUES (?, ?, ?)",
            (session["id"], filename, label))
        conn.commit()
        conn.close()
    except Exception as e:
        print("DB Error:", e)
    return render_template("next.html", label=label, image=filename)

@app.route('/history')
def history():
    conn = sqlite3.connect("signup.db")
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM predictions WHERE userid=? ORDER BY id DESC", (session["id"],))
    data = cursor.fetchall()
    conn.close()
    return render_template("history.html", data=data, user=session["user"])


# ── RAG routes ───────────────────────────
@app.route("/new", methods=["POST"])
def new():
    _init_rag()
    label = request.form.get("label")
    if _rag_retriever is None:
        return render_template("new.html", answer="RAG not available (report.pdf missing)", label=label)
    try:
        query = f"{label} explanation symptoms risk prevention"
        retrieved_docs = _rag_retriever.invoke(query)
        context = "\n\n".join([d.page_content for d in retrieved_docs])
        final_prompt = _rag_prompt.format(context=context, question=query)
        response = _rag_llm.invoke(final_prompt)
        return render_template("new.html", answer=response.content, label=label)
    except Exception as e:
        print("Error:", e)
        flash("Prediction failed")
        return redirect(url_for("home"))

@app.route("/ask", methods=["POST", "GET"])
def ask():
    _init_rag()
    if request.method == "POST":
        data = request.get_json()
        user_query = data.get("message")
        if _rag_retriever is None:
            return {"reply": "RAG not available (report.pdf missing)"}
        try:
            retrieved_docs = _rag_retriever.invoke(user_query)
            context = "\n\n".join([d.page_content for d in retrieved_docs])
            final_prompt = _rag_prompt.format(context=context, question=user_query)
            response = _rag_llm.invoke(final_prompt)
            return {"reply": response.content}
        except Exception as e:
            print(e)
            return {"reply": "Error processing request"}
    return render_template("chatbot.html", user=session["user"])


# ── PDF-upload chat ──────────────────────
@app.route("/upload-pdf", methods=["POST"])
def upload_pdf():
    global vectorstore1
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    files = request.files.getlist("file")
    all_docs = []
    for f in files:
        if f.filename == "":
            continue
        filename = secure_filename(f.filename)
        path = os.path.join(UPLOAD_FOLDER, filename)
        f.save(path)
        ext = filename.split(".")[-1].lower()
        if ext == "pdf":
            loader = PyPDFLoader(path)
            documents = loader.load()
        elif ext in ["png", "jpg", "jpeg"]:
            extracted_text = extract_text_from_image(path)
            pdf_path = os.path.join(UPLOAD_FOLDER, filename.rsplit(".", 1)[0] + ".pdf")
            create_pdf_from_text(extracted_text, pdf_path)
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
        else:
            continue
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        all_docs.extend(splitter.split_documents(documents))
    if not all_docs:
        return jsonify({"error": "No valid document content"}), 400
    vectorstore1 = load_or_create_store(all_docs)
    return jsonify({"message": "Files uploaded & indexed successfully"})

@app.route("/chat", methods=["POST"])
def chat():
    global vectorstore1
    if vectorstore1 is None:
        vectorstore1 = load_store_if_exists()
    if vectorstore1 is None:
        return jsonify({"error": "Upload PDF first"}), 400
    data = request.get_json(force=True)
    user_query = (data.get("question") or "").strip()
    if not user_query:
        return jsonify({"error": "Empty question"}), 400
    retriever = vectorstore1.as_retriever(search_kwargs={"k": 4})
    retrieved_docs = retriever.invoke(user_query)
    context = "\n\n".join([d.page_content for d in retrieved_docs])
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0,
                                  google_api_key=GOOGLE_API_KEY)
    prompt = ChatPromptTemplate.from_template("""
You are an intelligent document assistant.
Answer ONLY using the provided context. If not found say "Information not available."

Context: {context}
Question: {question}

Give: 1. Summary 2. Key Points 3. Explanation 4. Additional Notes
""")
    response = llm.invoke(prompt.format(context=context, question=user_query))
    return jsonify({"reply": response.content})


# ── Report downloads ─────────────────────
@app.route("/download_report", methods=["POST"])
def download_report():
    _init_rag()
    if "id" not in session:
        flash("Login required")
        return redirect(url_for("login"))
    if _rag_retriever is None:
        flash("RAG not available")
        return redirect(url_for("home"))
    label = request.form.get("label")
    filename = request.form.get("image")
    query = f"{label} explanation symptoms risk prevention"
    retrieved_docs = _rag_retriever.invoke(query)
    context = "\n\n".join([d.page_content for d in retrieved_docs])
    final_prompt = _rag_prompt.format(context=context, question=query)
    answer = _rag_llm.invoke(final_prompt).content
    conn = sqlite3.connect("signup.db")
    cursor = conn.cursor()
    cursor.execute("SELECT name, age, gender, email FROM users WHERE id=?", (session["id"],))
    user = cursor.fetchone()
    conn.close()
    name, age, gender, email = user if user else ("Unknown", "-", "-", "-")
    buffer = _build_report_pdf(label, filename, answer, name, age, gender, email)
    return send_file(buffer, as_attachment=True,
                     download_name="AI_Skin_Report.pdf", mimetype="application/pdf")

@app.route("/download_report_admin", methods=["POST"])
def download_report_admin():
    _init_rag()
    if "admin" not in session:
        flash("Admin login required")
        return redirect(url_for("login"))
    if _rag_retriever is None:
        flash("RAG not available")
        return redirect(url_for("login"))
    userid = request.form.get("userid")
    label = request.form.get("label")
    filename = request.form.get("image")
    query = f"{label} explanation symptoms risk prevention"
    retrieved_docs = _rag_retriever.invoke(query)
    context = "\n\n".join([d.page_content for d in retrieved_docs])
    final_prompt = _rag_prompt.format(context=context, question=query)
    answer = _rag_llm.invoke(final_prompt).content
    conn = sqlite3.connect("signup.db")
    cursor = conn.cursor()
    cursor.execute("SELECT name, age, gender, email FROM users WHERE id=?", (userid,))
    user = cursor.fetchone()
    conn.close()
    name, age, gender, email = user if user else ("Unknown", "-", "-", "-")
    buffer = _build_report_pdf(label, filename, answer, name, age, gender, email)
    return send_file(buffer, as_attachment=True,
                     download_name=f"Report_User_{userid}.pdf", mimetype="application/pdf")


# ── Admin ────────────────────────────────
@app.route("/admin")
def admin_dashboard():
    if "admin" not in session:
        return redirect(url_for("login"))
    conn = sqlite3.connect("signup.db")
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM users")
    total_users = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM predictions")
    total_predictions = cursor.fetchone()[0]
    conn.close()
    return render_template("admin_dashboard.html",
                            total_users=total_users, total_predictions=total_predictions)

@app.route("/admin/users")
def admin_users():
    if "admin" not in session:
        return redirect(url_for("login"))
    conn = sqlite3.connect("signup.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, email, age, gender FROM users")
    users = cursor.fetchall()
    conn.close()
    return render_template("admin_users.html", users=users)

@app.route("/admin/history")
def admin_history():
    if "admin" not in session:
        return redirect(url_for("login"))
    conn = sqlite3.connect("signup.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT predictions.id, users.name, predictions.filename,
               predictions.label, predictions.created_at, predictions.userid
        FROM predictions
        JOIN users ON predictions.userid = users.id
        ORDER BY predictions.id DESC""")
    data = cursor.fetchall()
    conn.close()
    return render_template("admin_history.html", data=data)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
