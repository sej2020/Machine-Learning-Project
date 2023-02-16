from email.message import EmailMessage
import smtplib

sender = "friendlyneighborhoodbot1879@gmail.com"
recipient = "jmelms@iu.edu"
message = "Hello world!"

email = EmailMessage()
email["From"] = sender
email["To"] = recipient
email["Subject"] = "Sent from Python!"
email.set_content(message)

smtp = smtplib.SMTP_SSL("smtp.gmail.com", 465)
smtp.login(sender, "vvwflugftkomguii")
smtp.sendmail(sender, recipient, email.as_string())
smtp.quit()