import os
from celery import Celery

celery = Celery(broker=os.environ["CELERY_BROKER_URL"], backend=os.environ["CELERY_BACKEND_URL"])


@celery.task(acks_late=True, task_track_started=True, name="process_file")
def process(unique_filename):
    pass
