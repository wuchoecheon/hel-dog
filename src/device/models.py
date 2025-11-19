from sqlalchemy import Column, String, ForeignKey

from src.database import Base

class Device(Base):
    __tablename__ = "device"

    device_id = Column(String, primary_key=True, unique=True, index=True)
    owner = Column(String, ForeignKey("user.email", ondelete="CASCADE"))