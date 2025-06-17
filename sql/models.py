from sqlalchemy import Column, Integer, String
from .database import Base

class News(Base):
    __tablename__ = "news"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String)
    summary = Column(String)
    content = Column(String)
    source = Column(String)
    link = Column(String)