🏍️ Helmet Violation & Accident Detection System
📖 Project Overview

This project is an AI-powered system capable of detecting helmet violations and road accidents from images and videos. The system stores detection results (with metadata) in the cloud, allows intelligent querying through an Agent-based RAG chatbot, and sends real-time alerts via Telegram bot, in addition to generating periodic reports via email.

✨ Skills Learned

🖼️ Computer Vision model training and fine-tuning with YOLO

🎯 Multi-class object detection: helmet, no-helmet, accident

📹 Image/video processing and inference pipeline design

💻 Streamlit-based web application development

☁️ Integration with AWS services:

🗂️ S3 for file storage

🗄️ RDS for structured logs

🖥️ EC2 for app hosting

🧠 RAG (Retrieval-Augmented Generation) for chatbot queries

🤖 Agent-based chatbot architecture with tool usage

📢 Real-time alerts with Telegram Bot API

📧 Email report generation and delivery using LLM agents

🚨 Problem Statement

Build an AI system to:

Detect helmet violations and accidents

Store metadata in the cloud

Provide intelligent queries via a chatbot

Send real-time alerts via Telegram

Generate reports for stakeholders

💼 Business Use Cases

⚖️ Traffic Law Enforcement: Detect and report helmet law violations automatically

🚑 Road Safety Monitoring: Real-time accident alerts for emergency responders

🌆 Smart City Systems: Integrate with CCTV networks for traffic analytics

📝 Insurance & Investigation: Provide proof images for accident claims

📊 Public Awareness Campaigns: Data-driven helmet usage and accident trends

🛠️ Approach
1️⃣ Model Preparation

Collect and annotate datasets:

Classes: no_helmet, accident

Train YOLO models locally or fine-tune pre-trained models

Export trained weights (best.pt) for deployment

2️⃣ Streamlit Application

Upload image/video for detection

Run inference on EC2 using trained YOLO model

Display results with bounding boxes

Save snapshots to AWS S3

Store metadata in AWS RDS (PostgreSQL + pgvector for RAG):

Timestamp, camera/location, class label, confidence, bounding box, S3 proof link

3️⃣ Accident Alert System

If class = accident:

Save image to S3 (detections/accident/)

Create RDS log entry

Send Telegram alert with timestamp, location, confidence, and image link

4️⃣ Agent-Based RAG Chatbot

Ingest detection logs into a vector store (pgvector or FAISS)

Chatbot tools:

🗃️ SQL tool for structured log queries

🔍 Vector search tool for semantic matching

☁️ S3 tool for fetching signed URLs

📄 Report tool for PDF/HTML summaries

📧 Email tool for sending reports

Example queries:

“Show all accidents from last week.”

“Which camera has the most helmet violations?”

“Email me a report of today’s detections.”

5️⃣ Reporting & Email

Users can trigger report generation from chatbot or UI

Reports include statistics, charts, and links to S3 images

✅ Results

🖼️ YOLO model detecting helmet, no-helmet, and accident cases

💻 Streamlit web app for uploads, detection, and queries

🗄️ AWS RDS database storing logs with metadata

☁️ AWS S3 bucket storing media and detection snapshots

🤖 Agent-based RAG chatbot answering safety questions

📢 Real-time Telegram alerts for accidents

📧 Email reporting system for stakeholders

📊 Evaluation Metrics

🎯 Model Performance: mAP, Precision, Recall per class

✅ Detection Accuracy: % correct detections

⏱️ Alert Timeliness: Time between detection and Telegram alert

🧠 Query Accuracy: Chatbot responses correctness

🔧 System Reliability: EC2 uptime, error rate of uploads

🗃️ Data Set

Sources:

Public helmet detection datasets from Kaggle

Custom collections from Roboflow

Traffic CCTV accident footage

Format: Images/videos (JPG, PNG, MP4) with YOLO annotation TXT files

Variables: Class label (helmet, no_helmet, accident), bounding box coordinates

Preprocessing: Resize, normalize, convert annotations to YOLO format

🎁 Project Deliverables

🖥️ YOLO training/fine-tuning code

📦 Trained model weights (best.pt)

💻 Streamlit app source code

🗄️ AWS RDS schema and sample data

☁️ S3 bucket structure documentation

📢 Telegram bot alert code

🤖 Chatbot agent implementation

📄 Sample PDF/HTML reports

📌 Project Guidelines

🐍 Follow PEP8 for Python code formatting

🔐 Store AWS credentials in environment variables

🔒 Keep S3 objects private; use pre-signed URLs for sharing

⚡ Optimize YOLO inference for EC2

🗂️ Use Git for version control; commit regularly

🧪 Test Telegram alerts in staging before production

🏷️ Technical Tags

YOLO, Computer Vision, Object Detection, Streamlit, AWS EC2, AWS RDS, AWS S3, RAG, LLM Agents, Telegram Bot, Email Automation, Traffic Law Enforcement, Accident Detection
