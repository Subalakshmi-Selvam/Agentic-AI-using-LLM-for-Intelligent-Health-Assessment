	import os
import urllib.request
from flask import *
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from werkzeug.utils import secure_filename
import sqlite3

# LangChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
import sqlite3, io, os, re, random
from datetime import datetime

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])



retriever = None
llm = None

def init_rag():
    global retriever, llm
    if retriever is not None:
        return
    print("Loading Cancer PDF...")
    if not os.path.exists("report.pdf"):
        print("WARNING: report.pdf not found.")
        return
    loader = PyPDFLoader("report.pdf")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=GOOGLE_API_KEY
    )

prompt = ChatPromptTemplate.from_template("""
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
def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
model1 = load_model('skin_disease_model.h5')
INDEX_FOLDER = "faiss_index"
os.makedirs(INDEX_FOLDER, exist_ok=True)
# 🔥 GLOBAL STORE
vectorstore1 = None


# ==========================
# 🔧 HELPERS
# ==========================
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

import google.generativeai as genai
genai.configure(api_key=GOOGLE_API_KEY)
def create_pdf_from_text(text, output_path):
    c = canvas.Canvas(output_path, pagesize=letter)

    width, height = letter
    y = height - 40

    lines = text.split("\n")

    for line in lines:
        c.drawString(40, y, line[:100])

        y -= 20

        if y < 40:
            c.showPage()
            y = height - 40

    c.save()


# =========================
# EXTRACT TEXT FROM IMAGE
# =========================
def extract_text_from_image(image_path):
    model = genai.GenerativeModel("gemini-2.5-flash")

    uploaded_file = genai.upload_file(image_path)

    response = model.generate_content([
        "Extract all text from this image accurately.",
        uploaded_file
    ])

    return response.text
def load_or_create_store(docs):
    embeddings = get_embeddings()

    if os.path.exists(os.path.join(INDEX_FOLDER, "index.faiss")):
        vs = FAISS.load_local(
            INDEX_FOLDER,
            embeddings,
            allow_dangerous_deserialization=True
        )
        vs.add_documents(docs)
    else:
        vs = FAISS.from_documents(docs, embeddings)

    vs.save_local(INDEX_FOLDER)
    return vs


def load_store_if_exists():
    embeddings = get_embeddings()

    if os.path.exists(os.path.join(INDEX_FOLDER, "index.faiss")):
        return FAISS.load_local(
            INDEX_FOLDER,
            embeddings,
            allow_dangerous_deserialization=True
        )
    return None


# ==========================
# 📤 UPLOAD PDF
# ==========================
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

        # =====================================
        # PDF FILE
        # =====================================
        if ext == "pdf":

            loader = PyPDFLoader(path)

            documents = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=150
            )

            docs = splitter.split_documents(documents)

            all_docs.extend(docs)

        # =====================================
        # IMAGE FILE
        # =====================================
        elif ext in ["png", "jpg", "jpeg"]:

            # Extract text using Gemini
            extracted_text = extract_text_from_image(path)

            # Convert text into PDF
            pdf_filename = filename.rsplit(".", 1)[0] + ".pdf"

            pdf_path = os.path.join(UPLOAD_FOLDER, pdf_filename)

            create_pdf_from_text(
                extracted_text,
                pdf_path
            )

            # Load created PDF
            loader = PyPDFLoader(pdf_path)

            documents = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=150
            )

            docs = splitter.split_documents(documents)

            all_docs.extend(docs)

    if not all_docs:
        return jsonify({
            "error": "No valid document content"
        }), 400

    vectorstore1 = load_or_create_store(all_docs)

    return jsonify({
        "message": "Files uploaded & indexed successfully"
    })
# ==========================
# 💬 CHAT (YOUR METHOD)
# ==========================
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

    # 🔍 Step 1: Retrieve docs
    retrieved_docs = retriever.invoke(user_query)

    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # 🤖 Step 2: LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=GOOGLE_API_KEY
    )

    # 🧠 Step 3: Prompt
    prompt = ChatPromptTemplate.from_template("""
You are an intelligent document assistant.

STRICT RULES:
- Answer ONLY using the provided context.
- If not found → "Information not available in the document."
- Do NOT hallucinate.

Context:
{context}

Question:
{question}

Give:
1. Summary
2. Key Points
3. Explanation
4. Additional Notes
""")

    final_prompt = prompt.format(
        context=context,
        question=user_query
    )

    # 🚀 Step 4: Generate answer
    response = llm.invoke(final_prompt)
    answer = response.content
    print(answer)

    return jsonify({"reply": answer})
@app.route("/chatupload")
def chatupload():
    return render_template("aichat.html",user=session["user"])

@app.route("/download_report", methods=["POST"])
def download_report():

    if "id" not in session:
        flash("Login required")
        return redirect(url_for("login"))
    label = request.form.get("label")
    filename = request.form.get("image")
     # 🔹 RAG QUERY
    query = f"{label} explanation symptoms risk prevention"

    retrieved_docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    final_prompt = prompt.format(
        context=context,
        question=query
    )

    response = llm.invoke(final_prompt)
    answer = response.content
    
    print(answer)

    # =========================
    # 🔹 GET USER DATA
    # =========================
    conn = sqlite3.connect("signup.db")
    cursor = conn.cursor()

    cursor.execute("SELECT name, age, gender, email FROM users WHERE id=?", (session["id"],))
    user = cursor.fetchone()
    conn.close()

    if user:
        name, age, gender, email = user
    else:
        name, age, gender, email = "Unknown", "-", "-", "-"

    # =========================
    # 🔹 PDF BUFFER
    # =========================
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)

    styles = getSampleStyleSheet()

    # -------- STYLES --------
    title_style = ParagraphStyle(
        name="Title",
        fontSize=20,
        alignment=TA_CENTER,
        textColor=colors.darkblue,
        spaceAfter=10
    )

    section_style = ParagraphStyle(
        name="Section",
        fontSize=13,
        textColor=colors.black,
        spaceAfter=6
    )

    normal = styles["Normal"]

    content = []

    # =========================
    # 🔹 TITLE
    # =========================
    content.append(Paragraph("<b>AI SKIN DIAGNOSIS REPORT</b>", title_style))

    line = Table([[""]], colWidths=[450])
    line.setStyle(TableStyle([
        ('LINEBELOW', (0,0), (-1,-1), 2, colors.orange)
    ]))
    content.append(line)
    content.append(Spacer(1, 15))

    # =========================
    # 🔹 PATIENT INFO
    # =========================
    report_id = "REP" + str(random.randint(1000, 9999))
    today = datetime.now().strftime("%d-%m-%Y")

    content.append(Paragraph("<b>Patient Information</b>", section_style))

    patient_table = Table([
        ["Name:", name],
        ["Email:", email],
        ["Age:", str(age)],
        ["Gender:", gender],
        ["Report ID:", report_id],
        ["Date:", today]
    ], colWidths=[120, 300])

    patient_table.setStyle(TableStyle([
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))

    content.append(patient_table)
    content.append(Spacer(1, 15))

    # =========================
    # 🔹 IMAGE
    # =========================
    image_path = os.path.join("static/uploads", filename)

    if os.path.exists(image_path):
        content.append(Paragraph("<b>Uploaded Image</b>", section_style))
        content.append(Image(image_path, width=160, height=160))
        content.append(Spacer(1, 15))

    # =========================
    # 🔹 PARSE LLM OUTPUT
    # =========================
    answer = answer.replace("**", "")

    def get_section(name):
        match = re.search(rf"{name}:(.*?)(\n\d\.|$)", answer, re.S)
        return match.group(1).strip() if match else "Not available"

    explanation = get_section("Explanation")
    symptoms = get_section("Symptoms")
    risk = get_section("Risk Factors")
    prevention = get_section("Prevention")
    notes = get_section("Additional Notes")

    # =========================
    # 🔹 AI DETAILS
    # =========================
    content.append(Paragraph("<b>AI Diagnosis</b>", section_style))
    content.append(Paragraph(f"Disease: {label}", normal))
    content.append(Spacer(1, 10))

    # =========================
    # 🔹 SECTIONS
    # =========================
    def add_section(title, text):
        content.append(Paragraph(f"<b>{title}</b>", section_style))
        content.append(Paragraph(text, normal))
        content.append(Spacer(1, 10))

    add_section("Explanation", explanation)
    add_section("Symptoms", symptoms)
    add_section("Risk Factors", risk)
    add_section("Prevention", prevention)
    add_section("Additional Notes", notes)

    # =========================
    # 🔹 BUILD PDF
    # =========================
    doc.build(content)
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="AI_Skin_Report.pdf",
        mimetype="application/pdf"
    )
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
    

@app.route('/store', methods=['POST'])
def upload_image():

    # 🔒 Check login
    if "id" not in session:
        flash("Please login first")
        return redirect(url_for("login"))

    if 'file' not in request.files:
        flash('No file uploaded')
        return redirect(url_for("home"))

    file = request.files['file']

    if file.filename == '':
        flash('No selected file')
        return redirect(url_for("home"))

    if not allowed_file(file.filename):
        flash('Invalid file type')
        return redirect(url_for("home"))

    from werkzeug.utils import secure_filename
    import uuid

    # ✅ Unique filename (IMPORTANT)
    filename = str(uuid.uuid4()) + "_" + secure_filename(file.filename)

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # 🧠 Prediction
    label = predict(file_path)

    try:
        conn = sqlite3.connect("signup.db")
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO predictions (userid, filename, label)
            VALUES (?, ?, ?)
        """, (session["id"], filename, label))

        conn.commit()
        conn.close()

    except Exception as e:
        print("DB Error:", e)
        flash("Error saving prediction")

    return render_template(
        "next.html",
        label=label,
        image=filename
    )

@app.route('/history')
def history():
    conn = sqlite3.connect("signup.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions where userid=? ORDER BY id DESC",(session["id"],))
    data = cursor.fetchall()
    conn.close()
    return render_template("history.html", data=data,user=session["user"])
@app.route("/new", methods=["POST"])
def new():
    try:
        label = request.form.get("label")

        # 🔹 RAG QUERY
        query = f"{label} explanation symptoms risk prevention"

        retrieved_docs = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        final_prompt = prompt.format(
            context=context,
            question=query
        )

        response = llm.invoke(final_prompt)
        answer = response.content

        return render_template("new.html", answer=answer, label=label)

    except Exception as e:
        print("Error:", e)
        flash("Prediction failed")
        return redirect(url_for("home"))
@app.route("/upload")
def upload():
    return render_template("home.html",user=session["user"])
@app.route("/chatbot")
def chatbot():
    return render_template("chatbot.html",user=session["user"])

@app.route("/ask", methods=["POST","GET"])
def ask():
    if request.method=="POST":
        data = request.get_json()
        user_query = data.get("message")

        try:
            # 🔹 Retrieve context
            retrieved_docs = retriever.invoke(user_query)
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])

            final_prompt = prompt.format(
                context=context,
                question=user_query
            )

            response = llm.invoke(final_prompt)
            answer = response.content
            print(answer)

            return {"reply": answer}

        except Exception as e:
            print(e)
            return {"reply": "Error processing request"}
    return render_template("chatbot.html",user=session["user"])


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
            cursor.execute("""
                INSERT INTO users (name, email, age, gender, password)
                VALUES (?, ?, ?, ?, ?)
            """, (name, email, age, gender, password))
            conn.commit()
            conn.close()

            flash("Registration Successful! Please login.", "success")
            return redirect(url_for("login"))

        except:
            flash("Email already exists!", "danger")

    return redirect("/login")
@app.route("/signin", methods=["GET", "POST"])
def signin():
    print("working")
    if request.method == "POST":
        username = request.form.get("user")
        password = request.form.get("password")

        if username == "admin" and password == "admin":
            session["admin"] = True
            return redirect(url_for("admin_dashboard"))

        # 🔹 Database check
        con = sqlite3.connect("signup.db")
        cur = con.cursor()

        cur.execute(
            "SELECT id,name, password FROM users WHERE email=? AND password=?",
            (username, password)
        )
        data = cur.fetchone()
        con.close()
        session["id"] = data[0]
        session["user"] = data[1]

        if data:
            return redirect(url_for("home"))
        else:
            flash("Invalid username or password")
            return render_template("signin.html")

    # 🔹 GET request
    return render_template("signin.html")

@app.route('/home')
def home():
	return render_template('dashboard.html',user=session["user"])

@app.route("/admin")
def admin_dashboard():

    if "admin" not in session:
        return redirect(url_for("login"))

    conn = sqlite3.connect("signup.db")
    cursor = conn.cursor()

    # total users
    cursor.execute("SELECT COUNT(*) FROM users")
    total_users = cursor.fetchone()[0]

    # total predictions
    cursor.execute("SELECT COUNT(*) FROM predictions")
    total_predictions = cursor.fetchone()[0]

    conn.close()

    return render_template(
        "admin_dashboard.html",
        total_users=total_users,
        total_predictions=total_predictions
    )

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
               predictions.label, predictions.created_at,predictions.userid
        FROM predictions
        JOIN users ON predictions.userid = users.id
        ORDER BY predictions.id DESC
    """)

    data = cursor.fetchall()
    conn.close()

    return render_template("admin_history.html", data=data)

	

@app.route("/about")
def about():
    return render_template("about.html",user=session["user"])
@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        message = request.form.get("message")

        # You can store in DB or send email later
        print(name, email, message)

        return "Message Sent Successfully ✅"

    return render_template("contact.html", user=session["user"])

@app.route("/logout")
def logout():
    session.clear()   # 🔥 removes all session data
    return redirect(url_for("login"))  # redirect to login page
@app.route("/download_report_admin", methods=["POST"])
def download_report_admin():

    # 🔒 Admin check
    if "admin" not in session:
        flash("Admin login required")
        return redirect(url_for("login"))

    userid = request.form.get("userid")
    label = request.form.get("label")
    filename = request.form.get("image")
    print(userid)

    # =========================
    # 🔹 GENERATE LLM OUTPUT (RAG)
    # =========================
    query = f"{label} explanation symptoms risk prevention"

    retrieved_docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    final_prompt = prompt.format(
        context=context,
        question=query
    )

    response = llm.invoke(final_prompt)
    answer = response.content

    # =========================
    # 🔹 GET USER DATA (ADMIN SIDE)
    # =========================
    conn = sqlite3.connect("signup.db")
    cursor = conn.cursor()

    cursor.execute("SELECT name, age, gender, email FROM users WHERE id=?", (userid,))
    user = cursor.fetchone()
    conn.close()

    if user:
        name, age, gender, email = user
    else:
        name, age, gender, email = "Unknown", "-", "-", "-"

    # =========================
    # 🔹 PDF SETUP
    # =========================
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        name="Title",
        fontSize=20,
        alignment=TA_CENTER,
        textColor=colors.darkblue,
        spaceAfter=10
    )

    section_style = ParagraphStyle(
        name="Section",
        fontSize=13,
        spaceAfter=6
    )

    normal = styles["Normal"]
    content = []

    # =========================
    # 🔹 TITLE
    # =========================
    content.append(Paragraph("<b>AI SKIN DIAGNOSIS REPORT</b>", title_style))

    line = Table([[""]], colWidths=[450])
    line.setStyle(TableStyle([
        ('LINEBELOW', (0,0), (-1,-1), 2, colors.orange)
    ]))
    content.append(line)
    content.append(Spacer(1, 15))

    # =========================
    # 🔹 USER INFO
    # =========================
    report_id = "REP" + str(random.randint(1000, 9999))
    today = datetime.now().strftime("%d-%m-%Y")

    content.append(Paragraph("<b>Patient Information</b>", section_style))

    patient_table = Table([
        ["Name:", name],
        ["Email:", email],
        ["Age:", str(age)],
        ["Gender:", gender],
        ["Report ID:", report_id],
        ["Date:", today]
    ], colWidths=[120, 300])

    content.append(patient_table)
    content.append(Spacer(1, 15))

    # =========================
    # 🔹 IMAGE
    # =========================
    image_path = os.path.join("static/uploads", filename)

    if os.path.exists(image_path):
        content.append(Paragraph("<b>Uploaded Image</b>", section_style))
        content.append(Image(image_path, width=160, height=160))
        content.append(Spacer(1, 15))

    # =========================
    # 🔹 PARSE LLM OUTPUT
    # =========================
    answer = answer.replace("**", "")

    def get_section(name):
        match = re.search(rf"{name}:(.*?)(\n\d\.|$)", answer, re.S)
        return match.group(1).strip() if match else "Not available"

    explanation = get_section("Explanation")
    symptoms = get_section("Symptoms")
    risk = get_section("Risk Factors")
    prevention = get_section("Prevention")
    notes = get_section("Additional Notes")

    # =========================
    # 🔹 DIAGNOSIS
    # =========================
    content.append(Paragraph("<b>AI Diagnosis</b>", section_style))
    content.append(Paragraph(f"Disease: {label}", normal))
    content.append(Spacer(1, 10))

    # =========================
    # 🔹 SECTIONS
    # =========================
    def add_section(title, text):
        content.append(Paragraph(f"<b>{title}</b>", section_style))
        content.append(Paragraph(text, normal))
        content.append(Spacer(1, 10))

    add_section("Explanation", explanation)
    add_section("Symptoms", symptoms)
    add_section("Risk Factors", risk)
    add_section("Prevention", prevention)
    add_section("Additional Notes", notes)

    # =========================
    # 🔹 BUILD PDF
    # =========================
    doc.build(content)
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"Report_User_{userid}.pdf",
        mimetype="application/pdf"
    )
if __name__ == '__main__':
    app.run()
