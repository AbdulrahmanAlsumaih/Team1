import smtplib
import time
import subprocess
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Hardcoded values
EMAIL_ADDRESS = "saurabh88609@gmail.com"
EMAIL_PASSWORD = "**************"
TO_ADDRESS = "saurabhkumar933@gmail.com"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
PROCESS_NAME_OR_PID = "your_process_name_or_pid"
CHECK_INTERVAL = 4 * 3600

def send_email(subject, message):
    msg = MIMEMultipart()
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = TO_ADDRESS
    msg['Subject'] = subject
    msg.attach(MIMEText(message, 'plain'))

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.sendmail(EMAIL_ADDRESS, TO_ADDRESS, msg.as_string())
        print("Email sent successfully.")
    except Exception as e:
        print(f"Failed to send email: {e}")

def check_process():
    # Check if process is running by name or PID
    result = subprocess.run(
        ["pgrep", "-f", PROCESS_NAME_OR_PID], capture_output=True, text=True
    )
    return result.returncode == 0

def main():
    last_notification_time = 0

    while True:
        process_running = check_process()
        current_time = time.time()

        if not process_running:
            # Send immediate notification if process has stopped
            send_email("Process Alert", f"The process '{PROCESS_NAME_OR_PID}' has stopped.")
            break  # exit loop if the process is no longer running

        if current_time - last_notification_time >= CHECK_INTERVAL:
            # Send periodic status update
            send_email("Process Status Update", f"The process '{PROCESS_NAME_OR_PID}' is still running.")
            last_notification_time = current_time

        # Wait a minute before checking again
        time.sleep(60)

if __name__ == "__main__":
    main()
