# emailer.py
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email(subject: str, body: str, to_addresses: list[str],
               smtp_server="smtp.gmail.com", smtp_port=587,
               username: str = None, password: str = None):
    """
    Simple email sender using SMTP.
    - to_addresses: list of recipient emails
    - username/password: your SMTP credentials
    """
    if not username or not password:
        raise ValueError("Provide SMTP username and password")

    msg = MIMEMultipart()
    msg['From'] = username
    msg['To'] = ', '.join(to_addresses)
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    server = smtplib.SMTP(smtp_server, smtp_port)
    server.starttls()
    server.login(username, password)
    server.send_message(msg)
    server.quit()
