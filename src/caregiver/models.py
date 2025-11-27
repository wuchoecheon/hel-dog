from sqlalchemy import Column, String, ForeignKey, Boolean

from src.database import Base

class Caregiver(Base):
    __tablename__ = "care_giver"

    id = Column(primary_key=True, nullable=False)
    caregiver = Column(String, ForeignKey("user.email", ondelete="CASCADE"), index=True)
    ward = Column(String, ForeignKey("user.email", ondelete="CASCADE"), index=True)
    accepted = Column(Boolean, nullable=False, server_default="False")