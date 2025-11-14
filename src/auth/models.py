from sqlalchemy import Column, String

from src.database import Base

class User(Base):
    __tablename__ = "user"

    email = Column(String, primary_key=True, unique=True, index=True)
    password = Column(String)