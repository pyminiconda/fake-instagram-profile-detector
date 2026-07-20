import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from backend.config import SMTP_SERVER, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD, SMTP_SENDER, FRONTEND_ORIGIN

def send_reset_email(to_email: str, token: str) -> bool:
    """
    Sends a password reset email using the configured SMTP server.
    Returns True if successful, False otherwise.
    """
    msg = MIMEMultipart("alternative")
    msg["Subject"] = "InstaGuard - Password Reset Request"
    msg["From"] = SMTP_SENDER
    msg["To"] = to_email

    text_content = f"""
    Hello,

    We received a request to reset your password for your InstaGuard account.
    Please click the link below to reset your password:
    
    {FRONTEND_ORIGIN}/reset-password?token={token}

    This link will expire in 15 minutes.

    If you did not request a password reset, please ignore this email.

    Stay safe,
    The InstaGuard Team
    """

    html_content = f"""
    <html>
      <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px;">
        <h2 style="color: #1a73e8;">InstaGuard Password Reset</h2>
        <p>Hello,</p>
        <p>We received a request to reset your password for your InstaGuard account.</p>
        <p>Please click the button below to choose a new password:</p>
        <div style="text-align: center; margin: 30px 0;">
            <a href="{FRONTEND_ORIGIN}/reset-password?token={token}" 
               style="background-color: #1a73e8; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; font-weight: bold; font-size: 1rem;">
               Reset Password
            </a>
        </div>
        <p style="font-size: 0.9rem; color: #666;">Or copy and paste this link into your browser: <br/>
        <a href="{FRONTEND_ORIGIN}/reset-password?token={token}">{FRONTEND_ORIGIN}/reset-password?token={token}</a></p>
        <p style="font-size: 0.9rem; color: #666;">This link will expire in 15 minutes.</p>
        <hr style="border: none; border-top: 1px solid #eee; margin: 30px 0;">
        <p style="font-size: 0.8rem; color: #999;">If you did not request a password reset, please ignore this email.</p>
      </body>
    </html>
    """

    part1 = MIMEText(text_content, "plain")
    part2 = MIMEText(html_content, "html")

    msg.attach(part1)
    msg.attach(part2)

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.sendmail(SMTP_SENDER, to_email, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        print(f"[ERROR] Failed to send email to {to_email}: {e}")
        return False
