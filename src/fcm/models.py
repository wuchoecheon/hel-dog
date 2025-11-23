from sqlalchemy import Column, String, UUID, ForeignKey

from src.database import Base

class FCM(Base):
    __tablename__ = "fcm"

    uuid = Column(UUID, primary_key=True) 
    fcm_token = Column(String, nullable=False) 
    owner = Column(String, ForeignKey('user.email', ondelete="CASCADE")) 
    