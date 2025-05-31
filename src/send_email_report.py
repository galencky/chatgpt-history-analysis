import os
import smtplib
import mimetypes
from email.message import EmailMessage
from dotenv import load_dotenv

def send_email_report(
    output_dir="data",
    log_filename="logs.txt",
    usage_csv="model_usage_frequency.csv"
):
    """
    Send all PNGs in data/, model_usage_frequency.csv (in data/), and logs.txt (in data/).
    Credentials from .env (EMAIL_USER, EMAIL_PASS). Prompts user for recipient.
    """
    # --- Load env ---
    load_dotenv()
    sender = os.getenv("EMAIL_USER")
    password = os.getenv("EMAIL_PASS")

    if not sender or not password:
        print("❌ EMAIL_USER or EMAIL_PASS not set in .env")
        return

    # --- Prompt user for recipient ---
    recipient = os.getenv("EMAIL_TO")
    if not recipient or "@" not in recipient:
        print("❌ EMAIL_TO not set in .env or invalid.")
        return

    # --- Prepare explicit files to attach ---
    files_to_attach = []

    # Attach all PNGs in ./data (but only top level, no subfolders)
    for fname in os.listdir(output_dir):
        full_path = os.path.join(output_dir, fname)
        if fname.lower().endswith(".png") and os.path.isfile(full_path):
            files_to_attach.append(full_path)

    # Attach model_usage_frequency.csv (must be in ./data)
    usage_csv_path = os.path.join(output_dir, usage_csv)
    if os.path.isfile(usage_csv_path):
        files_to_attach.append(usage_csv_path)
    else:
        print(f"⚠️ Usage CSV not found: {usage_csv_path}")

    # Attach logs.txt from data/
    log_full_path = os.path.join(output_dir, log_filename)
    if os.path.isfile(log_full_path):
        files_to_attach.append(log_full_path)
    else:
        print(f"⚠️ Log file not found: {log_full_path}")

    # --- Compose Email ---
    msg = EmailMessage()
    msg["Subject"] = "ChatGPT History Analysis Report"
    msg["From"] = sender
    msg["To"] = recipient
    msg.set_content(
        "Hello,\n\n"
        "Please find attached your ChatGPT usage report:\n"
        "• Model usage frequency table (CSV)\n"
        "• Plots (.png) summarizing your usage\n"
        "• The processing log (logs.txt)\n\n"
        "Questions? Reply to this email!\n\n"
        "Best regards,\nChatGPT Analyzer"
    )

    # --- Attach files ---
    for path in files_to_attach:
        if not os.path.isfile(path):
            print(f"❌ File missing: {path}")
            continue
        ctype, encoding = mimetypes.guess_type(path)
        if ctype is None or encoding is not None:
            ctype = "application/octet-stream"
        maintype, subtype = ctype.split("/", 1)
        with open(path, "rb") as f:
            file_data = f.read()
        msg.add_attachment(
            file_data,
            maintype=maintype,
            subtype=subtype,
            filename=os.path.basename(path)
        )
        print(f"✅ Attached: {os.path.basename(path)}")

    # --- Send email via Gmail SMTP ---
    print("📤 Sending email via Gmail SMTP...")
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(sender, password)
            smtp.send_message(msg)
        print("✅ Email sent successfully!")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")

# === CLI mode for direct testing ===
if __name__ == "__main__":
    send_email_report()
