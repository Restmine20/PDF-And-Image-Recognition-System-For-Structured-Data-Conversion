import os
from minio import Minio

minio_client = Minio(
    endpoint=os.environ['MINIO_ENDPOINT'],
    access_key=os.environ['MINIO_ROOT_USER'],
    secret_key=os.environ['MINIO_ROOT_PASSWORD'],
    secure=False
)

if not minio_client.bucket_exists(os.environ["UPLOADS_BUCKET_NAME"]):
    minio_client.make_bucket(os.environ["UPLOADS_BUCKET_NAME"])

if not minio_client.bucket_exists(os.environ["RESULTS_BUCKET_NAME"]):
    minio_client.make_bucket(os.environ["RESULTS_BUCKET_NAME"])
