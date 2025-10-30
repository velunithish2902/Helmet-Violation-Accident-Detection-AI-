import streamlit as st
import cv2
import numpy as np
import boto3
import psycopg2
from datetime import datetime, timedelta
from ultralytics import YOLO
from dotenv import load_dotenv
import os
import re
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from fpdf import FPDF
from botocore.client import Config
import requests
from openai import OpenAI

# ------------------ Load environment variables ------------------
load_dotenv()

# ------------------ AWS S3 Setup ------------------
S3_BUCKET = os.getenv("S3_BUCKET")
s3_client = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION"),
    config=Config(signature_version='s3v4')
)

# ------------------ PostgreSQL Setup ------------------
conn = psycopg2.connect(
    host=os.getenv("RDS_HOST"),
    database=os.getenv("RDS_DB_NAME"),
    user=os.getenv("RDS_USER"),
    password=os.getenv("RDS_PASSWORD")
)
cursor = conn.cursor()

# ------------------ Email Setup ------------------
ALERT_EMAIL_TO = os.getenv("ALERT_EMAIL_TO").split(",")
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")

# ------------------ Telegram Setup ------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# ------------------ S3 Upload Function ------------------
def upload_to_s3(file_bytes, bucket_name, object_name):
    try:
        s3_client.put_object(Bucket=bucket_name, Key=object_name, Body=file_bytes)
        file_url = f"https://{bucket_name}.s3.{os.getenv('AWS_REGION')}.amazonaws.com/{object_name}"
        return file_url
    except Exception as e:
        print(f"S3 Upload failed: {e}")
        return None

# ------------------ Telegram Alert ------------------
def send_telegram_alert(message, image_path=None):
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                      data={"chat_id": TELEGRAM_CHAT_ID, "text": message})
        if image_path and os.path.exists(image_path):
            files = {"photo": open(image_path, "rb")} if image_path.endswith((".jpg", ".png")) else {"video": open(image_path, "rb")}
            api_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto" if image_path.endswith((".jpg", ".png")) else f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendVideo"
            data = {"chat_id": TELEGRAM_CHAT_ID, "caption": "Detection Proof"}
            requests.post(api_url, data=data, files=files)
    except Exception as e:
        print(f"Telegram alert failed: {e}")

# ------------------ Email Sending ------------------
def send_email(subject, body, recipients, attachment_path=None):
    try:
        msg = MIMEMultipart()
        msg['From'] = SMTP_USER
        msg['To'] = ", ".join(recipients)
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        if attachment_path and os.path.exists(attachment_path):
            with open(attachment_path, 'rb') as f:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f"attachment; filename={os.path.basename(attachment_path)}")
            msg.attach(part)

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.sendmail(SMTP_USER, recipients, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        print(f"Email send failed: {e}")
        return False

# ------------------ PDF Report Creation ------------------
def create_pdf_report(data, query_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Road Safety AI Report", ln=True, align='C')
    pdf.set_font("Arial", '', 12)
    pdf.ln(5)
    pdf.multi_cell(0, 8, f"Report generated for query: {query_text}")
    pdf.ln(5)

    # Summary section
    total = len(data)
    accidents = sum(1 for d in data if d['class_label'].lower() == 'accident')
    no_helmet = sum(1 for d in data if d['class_label'].lower() in ['no_helmet', 'without_helmet'])

    pdf.cell(0, 8, f"Total Detections: {total}", ln=True)
    pdf.cell(0, 8, f"Accidents: {accidents}", ln=True)
    pdf.cell(0, 8, f"Helmet Violations: {no_helmet}", ln=True)
    pdf.ln(5)

    # Table header
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(25, 8, "Date", 1)
    pdf.cell(22, 8, "Time", 1)
    pdf.cell(35, 8, "Camera", 1)
    pdf.cell(35, 8, "Class", 1)
    pdf.cell(20, 8, "Conf", 1)
    pdf.cell(60, 8, "S3 Link", 1)
    pdf.ln()

    # Table rows
    pdf.set_font("Arial", '', 11)
    for d in data:
        ts = str(d.get('timestamp', ''))
        if ' ' in ts:
            date_part, time_part = ts.split(' ', 1)
        else:
            date_part, time_part = ts, ''

        pdf.cell(25, 8, date_part[:10], 1)
        pdf.cell(22, 8, time_part[:8], 1)
        pdf.cell(35, 8, str(d.get('camera_location', '')), 1)
        pdf.cell(35, 8, str(d.get('class_label', '')), 1)
        pdf.cell(20, 8, f"{float(d.get('confidence', 0.0)):.2f}", 1)
        s3_link = str(d.get('s3_link', '-'))
        if len(s3_link) > 40:
            s3_link = s3_link[:37] + "..."
        pdf.cell(60, 8, s3_link, 1)
        pdf.ln()

    filename = f"RoadSafety_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf.output(filename)
    return filename

# ------------------ YOLO Model ------------------
model = YOLO(os.getenv("YOLO_MODEL_PATH"))

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="Road Safety AI Dashboard", layout="wide")
st.markdown("<h1 style='text-align:center;'>🚦 Road Safety AI Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-weight:bold;'>Drive Safe | Wear Helmet | Don’t Drink & Drive | Save Lives</p>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
#-----------IMAGE TABLE-------
with col1:
    st.markdown("### Upload Image for Detection")
    camera = st.selectbox("Select Camera", ["Camera_1", "Camera_2"])
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png"], key="image_upload")

    if uploaded_image and st.button("Run Image Detection"):
        with st.spinner("Processing image..."):
            try:
                # ---------------- Decode image ----------------
                file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                # ---------------- YOLO detection ----------------
                results = model(img)
                annotated_img = results[0].plot()
                st.image(annotated_img, channels="BGR", caption="Detection Result")

                # ---------------- Save locally ----------------
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                local_filename = f"detection_{timestamp}.jpg"
                cv2.imwrite(local_filename, annotated_img)

                # ---------------- Parse detections ----------------
                predictions = []
                accident_detected = False
                confidence_value = 0.0

                for box, cls, conf in zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf):
                    class_label = model.names[int(cls)]
                    normalized_label = class_label.lower().replace(" ", "_")
                    confidence = float(conf)
                    bbox = [round(x, 2) for x in box.tolist()]

                    predictions.append({
                        "class_label": class_label,
                        "confidence": confidence,
                        "bbox": bbox,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "camera_location": camera,
                        "s3_link": "-"
                    })

                    bbox_str = '{' + ','.join(map(str, bbox)) + '}'
                    cursor.execute("""
                        INSERT INTO detections (timestamp, camera_location, class_label, confidence, bbox_coordinates, s3_link)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (datetime.now(), camera, class_label, confidence, bbox_str, "-"))
                    conn.commit()

                    if normalized_label == "accident":
                        accident_detected = True
                        confidence_value = confidence

                # ---------------- Show results ----------------
                st.markdown("### Detection Summary")
                st.dataframe(predictions)

                # ---------------- Upload annotated image to S3 ----------------
                s3_key = f"detections/{camera}/{local_filename}"
                s3_url = upload_to_s3(open(local_filename, "rb").read(), S3_BUCKET, s3_key)

                if s3_url:
                    st.success(f"✅ Uploaded to S3: {s3_url}")
                    # Update DB with S3 link
                    cursor.execute("UPDATE detections SET s3_link = %s WHERE timestamp >= NOW() - INTERVAL '1 minute'", (s3_url,))
                    conn.commit()
                    for pred in predictions:
                        pred["s3_link"] = s3_url
                else:
                    st.warning("⚠️ S3 upload failed, continuing without S3 link...")
                    s3_url = "S3 upload failed"

                # ---------------- PDF report ----------------
                pdf_file = create_pdf_report(predictions, "Uploaded Image Detection")

                # ---------------- Send alerts ----------------
                if accident_detected:
                    message = (
                        f"🚨 Accident Detected!\n"
                        f"📍 Location: {camera}\n"
                        f"🕒 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                        f"🎯 Confidence: {confidence_value:.2f}\n"
                        f"🌐 S3 Link: {s3_url}"
                    )
                    send_telegram_alert(message, local_filename)
                    send_email(subject="Accident Alert", body=message, recipients=ALERT_EMAIL_TO, attachment_path=pdf_file)
                    st.success("🚨 Accident alert (with S3 link) sent via Telegram and Email!")
                else:
                    message = (
                        f"✅ Detection Completed\n"
                        f"📍 Location: {camera}\n"
                        f"🕒 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                        f"🌐 S3 Link: {s3_url}"
                    )
                    send_telegram_alert(message, local_filename)
                    send_email("🚦 Road Safety Alert", message, ALERT_EMAIL_TO, pdf_file)
                    st.info("📩 Detection completed — report sent via Email and Telegram (S3 link included).")

                # ---------------- Download option ----------------
                st.download_button("📄 Download Report PDF", open(pdf_file, "rb"), file_name=pdf_file)

            except Exception as e:
                st.error(f"⚠️ Image Processing Error: {e}")

        
                 # ------------------ VIDEO DETECTION ------------------
  # ------------------ VIDEO DETECTION ------------------
st.markdown("---")
st.markdown("### Upload Video for Detection")
uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi"], key="video_upload")

if uploaded_video and st.button("Run Video Detection"):
    with st.spinner("Processing video... This may take a while ⏳"):
        try:
            # Create a unique temp filename for each uploaded video
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            video_path = f"uploaded_{timestamp}.mp4"
            with open(video_path, "wb") as f:
                f.write(uploaded_video.read())

            # Run YOLO inference (unique output folder for each upload)
            output_folder = f"run_{timestamp}"
            results = model.predict(
                source=video_path,
                save=True,
                project="runs/video_results",
                name=output_folder,
                conf=0.4,
                stream=False
            )

            # Get annotated video path
            output_dir = f"runs/video_results/{output_folder}"
            output_path = None
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    if file.endswith((".mp4", ".avi")):
                        output_path = os.path.join(root, file)
                        break

            if not output_path:
                st.error("❌ Annotated video not found.")
                st.stop()

            st.video(output_path)

            # Analyze for detections
            cap = cv2.VideoCapture(output_path)
            detections = []
            accident_detected = False
            helmet_violation = False
            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                if frame_count % 15 != 0:
                    continue

                frame_results = model(frame)
                for r in frame_results:
                    for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                        class_label = model.names[int(cls)]
                        norm_label = class_label.lower().replace(" ", "_")
                        confidence = float(conf)
                        timestamp_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        detections.append({
                            "timestamp": timestamp_now,
                            "camera_location": camera,
                            "class_label": class_label,
                            "confidence": confidence
                        })

                        if norm_label == "accident":
                            accident_detected = True
                        elif norm_label in ["no_helmet", "without_helmet"]:
                            helmet_violation = True

            cap.release()

            # Upload annotated video to S3 (unique filename)
            s3_key = f"detections/videos/{os.path.basename(output_path)}"
            s3_client.upload_file(output_path, S3_BUCKET, s3_key)
            s3_url = f"https://{S3_BUCKET}.s3.amazonaws.com/{s3_key}"

            # Save detections to RDS
            for det in detections:
                cursor.execute("""
                    INSERT INTO detections (timestamp, camera_location, class_label, confidence, s3_link)
                    VALUES (%s, %s, %s, %s, %s)
                """, (det["timestamp"], det["camera_location"], det["class_label"], det["confidence"], s3_url))
            conn.commit()

            st.success("✅ Video processed and uploaded to S3!")
            st.markdown(f"[🔗 View Annotated Video in S3]({s3_url})")

            # 🚨 Send Telegram alerts (specific to this video)
            if accident_detected:
                alert_msg = (
                    f"🚨 Accident detected in video!\n"
                    f"📍 Location: {camera}\n"
                    f"🕒 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )
                send_telegram_alert(alert_msg, output_path)
                pdf_file = create_pdf_report(detections, "Accident Video Detection")
                send_email("Accident Video Alert", alert_msg, ALERT_EMAIL_TO, pdf_file)
                st.warning("🚨 Accident alert sent via Telegram and Email!")

            elif helmet_violation:
                alert_msg = (
                    f"⚠️ Helmet violation detected!\n"
                    f"📍 Location: {camera}\n"
                    f"🕒 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )
                send_telegram_alert(alert_msg, output_path)
                pdf_file = create_pdf_report(detections, "Helmet Violation Video Detection")
                send_email("Helmet Violation Alert", alert_msg, ALERT_EMAIL_TO, pdf_file)
                st.info("⚠️ Helmet violation alert sent via Telegram and Email!")

        except Exception as e:
            st.error(f"Video Processing Error: {e}")


# ------------------ POSTGRES QUERY ------------------
with col2:
    st.markdown("### Query Last Detections")
    example_queries = [
        "Show me all accidents from last week.",
        "Which camera has the most helmet violations?",
        "Email me a report of today's detections."
    ]

    if 'run_query' not in st.session_state:
        st.session_state['run_query'] = False

    st.markdown("**Example Queries:**")
    cols = st.columns(len(example_queries))
    for i, q in enumerate(example_queries):
        if cols[i].button(q, key=f"example_{i}"):
            st.session_state['user_query_temp'] = q
            st.session_state['run_query'] = True

    user_query = st.text_input(
        "Or enter your query here:",
        value=st.session_state.get('user_query_temp', '')
    )

    if st.button("Run Query") or st.session_state['run_query']:
        if user_query:
            try:
                def tool_postgres(query: str, default_limit: int = 10):
                    q = query.lower().strip()
                    sql_query = ""
                    params = []

                    # CASE 1: Helmet violation count
                    if "most helmet" in q or "helmet violations" in q or "count helmet" in q:
                        sql_query = """
                            SELECT camera_location, COUNT(*) AS helmet_violations
                            FROM detections
                            WHERE lower(class_label) IN (
                                'no_helmet',
                                'without_helmet',
                                'helmet_violation',
                                'helmet_violation_or_accident'
                            )
                            GROUP BY camera_location
                            ORDER BY helmet_violations DESC
                            LIMIT 1
                        """
                        cursor.execute(sql_query)
                        row = cursor.fetchone()
                        if row:
                            data = [{"camera_location": row[0], "helmet_violations": row[1]}]
                            answer = f"📸 Camera with most helmet violations: **{row[0]}** ({row[1]} violations)"
                        else:
                            data, answer = [], "No helmet violation data found."
                        return {"rows": data, "answer": answer}

                    # CASE 2: Time-based + filters
                    where = []
                    if "accident" in q:
                        where.append("lower(class_label) = 'accident'")
                    elif "helmet" in q:
                        where.append("lower(class_label) IN ('no_helmet', 'without_helmet', 'helmet_violation', 'helmet_violation_or_accident')")

                    if "today" in q:
                        start = datetime.now().date()
                        end = start + timedelta(days=1)
                        where.append("timestamp BETWEEN %s AND %s")
                        params += [start, end]
                    elif "yesterday" in q:
                        start = datetime.now().date() - timedelta(days=1)
                        end = start + timedelta(days=1)
                        where.append("timestamp BETWEEN %s AND %s")
                        params += [start, end]
                    elif "week" in q:
                        start = datetime.now().date() - timedelta(days=7)
                        where.append("timestamp >= %s")
                        params.append(start)

                    sql_query = """
                        SELECT id::text AS id, timestamp, camera_location, class_label, confidence, s3_link
                        FROM detections
                    """
                    if where:
                        sql_query += " WHERE " + " AND ".join(where)
                    sql_query += " ORDER BY timestamp DESC LIMIT %s"
                    params.append(default_limit)

                    cursor.execute(sql_query, tuple(params))
                    rows = cursor.fetchall()
                    cols = [desc[0] for desc in cursor.description]
                    data = [dict(zip(cols, row)) for row in rows]

                    answer = f"Retrieved {len(data)} detection records."
                    if "email" in q:
                        pdf_file = create_pdf_report(data, query)
                        send_email(
                            subject="Road Safety AI Report",
                            body=f"Attached is the report for your query: {query}",
                            recipients=ALERT_EMAIL_TO,
                            attachment_path=pdf_file
                        )
                        answer += " 📧 Report emailed successfully."

                    return {"rows": data, "answer": answer}

                resp = tool_postgres(user_query)
                st.markdown("### Query Result")
                st.write(resp["answer"])
                if resp["rows"]:
                    st.dataframe(resp["rows"])

            except Exception as e:
                st.error(f"Query Error: {e}")

        st.session_state['run_query'] = False
   # =========================================================
#  AGENT-BASED RAG LLM CHATBOT (RAG + Tools)
# =========================================================
st.markdown("---")
st.markdown("## 🤖 Agent-Based RAG Chatbot")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ------------------ TOOL: POSTGRES STRUCTURED QUERY ------------------
def tool_postgres(query: str, default_limit: int = 10):
    q = query.lower().strip()
    sql_query = ""
    params = []

    # CASE 1: Helmet violation count
    if "most helmet" in q or "helmet violations" in q or "count helmet" in q:
        sql_query = """
            SELECT camera_location, COUNT(*) AS helmet_violations
            FROM detections
            WHERE lower(class_label) IN (
                'no_helmet',
                'without_helmet',
                'helmet_violation',
                'helmet_violation_or_accident'
            )
            GROUP BY camera_location
            ORDER BY helmet_violations DESC
            LIMIT 1
        """
        cursor.execute(sql_query)
        row = cursor.fetchone()
        if row:
            data = [{"camera_location": row[0], "helmet_violations": row[1]}]
            answer = f"📸 Camera with most helmet violations: **{row[0]}** ({row[1]} violations)"
        else:
            data, answer = [], "No helmet violation data found."
        return {"rows": data, "answer": answer}

    # CASE 2: Time-based filters
    where = []
    if "accident" in q:
        where.append("lower(class_label) = 'accident'")
    elif "helmet" in q:
        where.append("lower(class_label) IN ('no_helmet', 'without_helmet', 'helmet_violation', 'helmet_violation_or_accident')")

    if "today" in q:
        start = datetime.now().date()
        end = start + timedelta(days=1)
        where.append("timestamp BETWEEN %s AND %s")
        params += [start, end]
    elif "yesterday" in q:
        start = datetime.now().date() - timedelta(days=1)
        end = start + timedelta(days=1)
        where.append("timestamp BETWEEN %s AND %s")
        params += [start, end]
    elif "week" in q:
        start = datetime.now().date() - timedelta(days=7)
        where.append("timestamp >= %s")
        params.append(start)

    sql_query = """
        SELECT id::text AS id, timestamp, camera_location, class_label, confidence, s3_link
        FROM detections
    """
    if where:
        sql_query += " WHERE " + " AND ".join(where)
    sql_query += " ORDER BY timestamp DESC LIMIT %s"
    params.append(default_limit)

    cursor.execute(sql_query, tuple(params))
    rows = cursor.fetchall()
    cols = [desc[0] for desc in cursor.description]
    data = [dict(zip(cols, row)) for row in rows]

    answer = f"Retrieved {len(data)} detection records."
    if "email" in q:
        pdf_file = create_pdf_report(data, query)
        send_email(
            subject="Road Safety AI Report",
            body=f"Attached is the report for your query: {query}",
            recipients=ALERT_EMAIL_TO,
            attachment_path=pdf_file
        )
        answer += " 📧 Report emailed successfully."

    return {"rows": data, "answer": answer}


# ------------------ TOOL: VECTOR STORE SEARCH ------------------
def vector_search_tool(query, top_k=5):
    try:
        emb = client.embeddings.create(model="text-embedding-3-small", input=query).data[0].embedding
        cursor.execute("""
            SELECT id, timestamp, camera_location, class_label, confidence, s3_link,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM detections
            ORDER BY similarity DESC
            LIMIT %s
        """, (emb, top_k))
        rows = cursor.fetchall()
        cols = [desc[0] for desc in cursor.description]
        return [dict(zip(cols, row)) for row in rows]
    except Exception as e:
        return [{"error": str(e)}]


# ------------------ AGENT ROUTER ------------------
def run_agent(user_query):
    q = user_query.lower().strip()

    decision_prompt = f"""
    You are a smart assistant with access to these tools:
    1. SQL tool: For structured log queries (accident count, helmet violations).
    2. Vector tool: For semantic queries (natural language questions).
    3. Email tool: To send report.
    4. Report tool: To create summary PDF.

    User query: "{user_query}"
    Which tool(s) should be used? Reply as JSON with keys: tool, reason.
    """

    decision = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Decide tool routing."},
            {"role": "user", "content": decision_prompt}
        ]
    )

    tool_choice = decision.choices[0].message.content
    st.write(f"🧠 Agent Tool Decision → {tool_choice}")

    if "sql" in tool_choice.lower():
        return tool_postgres(user_query)
    elif "vector" in tool_choice.lower():
        results = vector_search_tool(user_query)
        return {"rows": results, "answer": f"Semantic results ({len(results)} items)."}
    elif "email" in tool_choice.lower():
        data = tool_postgres(user_query)["rows"]
        pdf = create_pdf_report(data, user_query)
        send_email("Requested Report", "Attached is your requested detection report.", ALERT_EMAIL_TO, pdf)
        return {"rows": data, "answer": "📧 Report emailed successfully."}
    else:
        return {"rows": [], "answer": "No suitable tool detected."}


# ------------------ STREAMLIT CHAT UI ------------------
st.markdown("### 💬 Ask me anything about detections")
chat_query = st.text_input("Your question:", placeholder="e.g. Show me all accident detections this week")
if st.button("Ask AI"):
    with st.spinner("Thinking..."):
        response = run_agent(chat_query)
        st.write(response["answer"])
        if response["rows"]:
            st.dataframe(response["rows"])
