"""Optional AWS SNS alert (serverless notification). Silently skipped if not configured.

Setup:
  1. Create an SNS topic in AWS and subscribe your phone (SMS) or email.
  2. export VIGILEYE_SNS_TOPIC_ARN="arn:aws:sns:ap-south-1:123456789012:vigileye"
  3. export AWS_REGION="ap-south-1"   (and configure credentials with `aws configure`)
"""
import os
import threading
import time

COOLDOWN_SEC = 60
_last_sent = 0.0


def send_cloud_alert(message: str) -> None:
    global _last_sent
    topic = os.environ.get("VIGILEYE_SNS_TOPIC_ARN")
    now = time.time()
    if not topic or now - _last_sent < COOLDOWN_SEC:
        return
    _last_sent = now
    threading.Thread(target=_publish, args=(topic, message), daemon=True).start()


def _publish(topic: str, message: str) -> None:
    try:
        import boto3
        client = boto3.client("sns", region_name=os.environ.get("AWS_REGION", "ap-south-1"))
        client.publish(TopicArn=topic, Subject="VigilEye driver alert", Message=message)
        print("[cloud] alert sent")
    except Exception as exc:  # never crash the detector because of the cloud
        print(f"[cloud] alert failed: {exc}")