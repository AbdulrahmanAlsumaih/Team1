import smtplib
from email.mime.text import MIMEText
import subprocess
from datetime import datetime

def send_email_alert(subject, body=""):
    sender_email = "saurabhkumar933@gmail.com"
    receiver_email = "saurabhkumar933@gmail.com"
    password = "##########################"  # Replace with your app-specific password

    msg = MIMEText(body if body else "Script monitoring alert")
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = receiver_email

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
            print("Email alert sent successfully!")
    except Exception as e:
        print("Failed to send email alert:", e)

def check_script_running(script_name="my_train.py"):
    print(f"Checking if {script_name} is running...")
    result = subprocess.run(["pgrep", "-f", script_name], stdout=subprocess.PIPE)
    is_running = result.returncode == 0  # 0 means the script is running
    print(f"Script running status: {'Running' if is_running else 'Not running'}")
    return is_running

try:
    if not check_script_running("my_train.py"):
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{current_time} - Script not running, sending alert email.")
        send_email_alert(f"Script Stopped @ {current_time}", "The script my_train.py has stopped unexpectedly.")
except Exception as e:
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{current_time} - Error occurred, sending alert email.")
    send_email_alert(f"Script Error @ {current_time}", f"An error occurred:\n\n{str(e)}")
