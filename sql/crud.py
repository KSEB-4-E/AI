from sqlalchemy.orm import Session
from . import models, schemas

def get_latest_news(db: Session, limit: int = 30):
    return db.query(models.News).order_by(models.News.id.desc()).limit(limit).all()