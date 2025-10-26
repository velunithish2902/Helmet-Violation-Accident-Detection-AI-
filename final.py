import streamlit as st
import cv2
import numpy as np
import boto3
import psycopg2
from datetime import datetime
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

# ------------------ Helper Functions ------------------
def send_telegram_alert(message, image_path=None):
    """Send Telegram alert with optional image/video."""
    try:
        # Message
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                      data={"chat_id": TELEGRAM_CHAT_ID, "text": message})
        # Image/Video
        if image_path and os.path.exists(image_path):
            files = {"photo": open(image_path, "rb")} if image_path.endswith((".jpg", ".png")) else {"video": open(image_path, "rb")}
            api_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto" if image_path.endswith((".jpg", ".png")) else f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendVideo"
            data = {"chat_id": TELEGRAM_CHAT_ID, "caption": "Detection Proof"}
            requests.post(api_url, data=data, files=files)
    except Exception as e:
        print(f"Telegram alert failed: {e}")

def send_email(subject, body, recipients, attachment_path=None):
    """Send email with optional attachment."""
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

def create_pdf_report(data, query_text):
    """Generate PDF report summarizing detections."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Road Safety AI Report", ln=True, align='C')
    pdf.set_font("Arial", '', 12)
    pdf.ln(5)
    pdf.multi_cell(0, 8, f"Report generated for query: {query_text}")
    pdf.ln(5)

    total = len(data)
    accidents = sum(1 for d in data if d['class_label'].lower() == 'accident')
    no_helmet = sum(1 for d in data if d['class_label'].lower() == 'no_helmet')

    pdf.cell(0, 8, f"Total Detections: {total}", ln=True)
    pdf.cell(0, 8, f"Accidents: {accidents}", ln=True)
    pdf.cell(0, 8, f"Helmet Violations: {no_helmet}", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(30, 8, "Timestamp", 1)
    pdf.cell(40, 8, "Camera", 1)
    pdf.cell(40, 8, "Class", 1)
    pdf.cell(20, 8, "Conf", 1)
    pdf.cell(60, 8, "S3 Link", 1)
    pdf.ln()

    pdf.set_font("Arial", '', 12)
    for d in data:
        pdf.cell(30, 8, str(d.get('timestamp', '')), 1)
        pdf.cell(40, 8, str(d.get('camera_location', '')), 1)
        pdf.cell(40, 8, str(d.get('class_label', '')), 1)
        pdf.cell(20, 8, str(round(d.get('confidence', 0.0), 2)), 1)
        pdf.cell(60, 8, str(d.get('s3_link', '-')), 1)
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

# ------------------ IMAGE DETECTION ------------------
with col1:
    st.markdown("### Upload Image for Detection")
    camera = st.selectbox("Select Camera", ["Camera_1", "Camera_2"])
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png"], key="image_upload")

    if uploaded_image and st.button("Run Image Detection"):
        with st.spinner("Processing image..."):
            try:
                file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                results = model(img)
                annotated_img = results[0].plot()
                st.image(annotated_img, channels="BGR", caption="Detection Result")

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                local_filename = f"detection_{timestamp}.jpg"
                cv2.imwrite(local_filename, annotated_img)

                predictions = []
                save_to_s3 = False
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
                        "s3_link": ""
                    })

                    bbox_str = '{' + ','.join(map(str, bbox)) + '}'
                    cursor.execute("""
                        INSERT INTO detections (timestamp, camera_location, class_label, confidence, bbox_coordinates, s3_link)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (datetime.now(), camera, class_label, confidence, bbox_str, ""))
                    conn.commit()

                    if normalized_label in ["no_helmet", "accident"]:
                        save_to_s3 = True
                        if normalized_label == "accident":
                            accident_detected = True
                            confidence_value = confidence

                if predictions:
                    st.markdown("### Detection Summary")
                    st.dataframe(predictions)

                if save_to_s3:
                    s3_key = f"detections/{local_filename}"
                    s3_client.upload_file(local_filename, S3_BUCKET, s3_key)
                    s3_url = f"https://{S3_BUCKET}.s3.amazonaws.com/{s3_key}"
                    st.success(f"Uploaded to S3: {s3_url}")

                    for pred in predictions:
                        pred["s3_link"] = s3_url

                    if accident_detected:
                        message = (
                            f"🚨 Accident Detected!\n"
                            f"📍 Location: {camera}\n"
                            f"🕒 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                            f"🎯 Confidence: {confidence_value:.2f}"
                        )
                        send_telegram_alert(message, local_filename)
                        pdf_file = create_pdf_report(predictions, "Accident Detection")
                        send_email(subject="Accident Alert", body=message, recipients=ALERT_EMAIL_TO, attachment_path=pdf_file)
                        st.success("Accident alert sent via Telegram and Email!")
                else:
                    st.info("✅ No 'no-helmet' or 'accident' detected.")

            except Exception as e:
                st.error(f"Image Processing Error: {e}")

# ------------------ VIDEO DETECTION ------------------
with col1:
    st.markdown("### Upload Video for Detection")
    uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi"], key="video_upload")

    if uploaded_video and st.button("Run Video Detection"):
        with st.spinner("Processing video..."):
            try:
                video_bytes = uploaded_video.read()
                video_path = f"uploaded_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                with open(video_path, "wb") as f:
                    f.write(video_bytes)

                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_idx = 0

                video_output_path = f"detection_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = None

                predictions = []
                save_to_s3 = False
                accident_detected = False
                confidence_value = 0.0

                progress_bar = st.progress(0)

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    results = model(frame)
                    annotated_frame = results[0].plot()

                    if out is None:
                        h, w, _ = annotated_frame.shape
                        out = cv2.VideoWriter(video_output_path, fourcc, 20, (w, h))
                    out.write(annotated_frame)

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
                            "s3_link": ""
                        })

                        if normalized_label in ["no_helmet", "accident"]:
                            save_to_s3 = True
                            if normalized_label == "accident":
                                accident_detected = True
                                confidence_value = confidence

                    frame_idx += 1
                    progress_bar.progress(frame_idx / total_frames)

                cap.release()
                out.release()
                st.video(video_output_path, format="video/mp4", start_time=0)
                st.success("Video processing completed!")

                if save_to_s3:
                    s3_key = f"detections/{os.path.basename(video_output_path)}"
                    s3_client.upload_file(video_output_path, S3_BUCKET, s3_key)
                    s3_url = f"https://{S3_BUCKET}.s3.amazonaws.com/{s3_key}"
                    st.success(f"Uploaded to S3: {s3_url}")

                    for pred in predictions:
                        pred["s3_link"] = s3_url

                    if accident_detected:
                        message = (
                            f"🚨 Accident Detected in Video!\n"
                            f"📍 Location: {camera}\n"
                            f"🕒 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                            f"🎯 Confidence: {confidence_value:.2f}"
                        )
                        send_telegram_alert(message, video_output_path)
                        pdf_file = create_pdf_report(predictions, "Accident Detection Video")
                        send_email(subject="Accident Alert (Video)", body=message, recipients=ALERT_EMAIL_TO, attachment_path=pdf_file)
                        st.success("Accident alert sent via Telegram and Email!")
                else:
                    st.info("✅ No 'no-helmet' or 'accident' detected in video.")

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

    user_query = st.text_input("Or enter your query here:", value=st.session_state.get('user_query_temp', ''))

    if st.button("Run Query") or st.session_state['run_query']:
        if user_query:
            try:
                def tool_postgres(query: str, default_limit: int = 10):
                    m = re.search(r"last\s+(\d+)\s*(\w+)?", query.lower())
                    limit = default_limit
                    class_filter = None
                    if m:
                        limit = int(m.group(1))
                        class_filter = m.group(2)

                    sql_query = "SELECT id::text AS id, timestamp, camera_location, class_label, confidence, s3_link FROM detections"
                    params = ()
                    if class_filter and class_filter not in ["detections", "all"]:
                        sql_query += " WHERE lower(class_label) = %s"
                        params = (class_filter,)
                    sql_query += " ORDER BY timestamp DESC LIMIT %s"
                    params += (limit,)

                    cursor.execute(sql_query, params)
                    rows = cursor.fetchall()
                    cols = [desc[0] for desc in cursor.description]
                    data = [{c: (v.isoformat() if hasattr(v, 'isoformat') else v) for c, v in zip(cols, row)} for row in rows]

                    summary = f"Retrieved {len(data)} records."
                    if "email" in query.lower():
                        pdf_file = create_pdf_report(data, query)
                        send_email(subject="Road Safety AI Report", body=f"Please find attached the PDF report for query: {query}", recipients=ALERT_EMAIL_TO, attachment_path=pdf_file)
                        summary += " PDF Report emailed to recipients."
                    return {"rows": data, "answer": summary}

                resp = tool_postgres(user_query)
                st.markdown("### Query Result")
                st.write(resp["answer"])
                if resp["rows"]:
                    st.dataframe(resp["rows"])
            except Exception as e:
                st.error(f"Query Error: {e}")

        st.session_state['run_query'] = False
