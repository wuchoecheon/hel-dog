from pyfcm import FCMNotification
from src.config import FCM_KEY, FIREBASE_PROJECT_ID

def send_notification(fcm_token:str, title, body):
    push_service = FCMNotification(FCM_KEY, FIREBASE_PROJECT_ID)
    return push_service.notify(
        fcm_token=fcm_token, 
        notification_title=title, 
        notification_body=body,
        notification_image=None,
    )
