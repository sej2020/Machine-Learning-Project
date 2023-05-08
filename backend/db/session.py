from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from configuration.config import settings

SQLALCHEMY_DATABASE_URL = settings.DATABASE_URL
engine = create_engine(SQLALCHEMY_DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db_actual():
    db = SessionLocal()
    return db

