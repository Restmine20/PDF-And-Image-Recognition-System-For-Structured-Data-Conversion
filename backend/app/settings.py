import os
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from sqlalchemy import create_engine, Column, Integer, String, DateTime, text, ForeignKey, Boolean
from minio import Minio
from flask_jwt_extended import JWTManager
import redis
from flask_mail import Mail

minio_client = Minio(
    endpoint=os.environ['MINIO_ENDPOINT'],
    access_key=os.environ['MINIO_ROOT_USER'],
    secret_key=os.environ['MINIO_ROOT_PASSWORD'],
    secure=False
)

global_minio_client = Minio(
    endpoint=os.environ['GLOBAL_MINIO_ENDPOINT'],
    access_key=os.environ['GLOBAL_MINIO_ROOT_USER'],
    secret_key=os.environ['GLOBAL_MINIO_ROOT_PASSWORD'],
    secure=True
)

if not minio_client.bucket_exists(os.environ["UPLOADS_BUCKET_NAME"]):
    minio_client.make_bucket(os.environ["UPLOADS_BUCKET_NAME"])
if not minio_client.bucket_exists(os.environ["RESULTS_BUCKET_NAME"]):
    minio_client.make_bucket(os.environ["RESULTS_BUCKET_NAME"])


engine = create_engine(
    os.environ["SQLALCHEMY_DATABASE_URI"],
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True
)

SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


jwt = JWTManager()
mail = Mail()


jwt_redis_blocklist = redis.from_url(os.environ["JWT_REDIS_BLACKLIST_URL"], decode_responses=True)
