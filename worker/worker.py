import openpyxl
import os
from JPGProcessingModule.main import main as jpg_main
from PDFProcessingModule.main import main as pdf_main
from celery import Celery
from settings import minio_client


celery = Celery(broker=os.environ["CELERY_BROKER_URL"], backend=os.environ["CELERY_BACKEND_URL"])


@celery.task(acks_late=True, task_track_started=True, name="process_file")
def process(unique_filename):

    output_filename, extension = os.path.splitext(unique_filename)
    extension = extension.lower()

    source_path = os.path.join(os.environ["TEMP_FOLDER"], unique_filename)
    result_filename = f"{output_filename}.xlsx"
    result_path = os.path.join(os.environ["TEMP_FOLDER"], result_filename)

    try:
        minio_client.fget_object(
            bucket_name=os.environ["UPLOADS_BUCKET_NAME"],
            object_name=unique_filename,
            file_path=source_path
        )

        workbook = openpyxl.Workbook()
        workbook.remove(workbook.worksheets[0])

        if extension == ".pdf":
            pdf_main(source_path, workbook)
        elif extension in (".png", ".jpg", ".jpeg"):
            jpg_main(source_path, workbook)
        else:
            raise ValueError("Wrong file type")

        workbook.save(result_path)

        minio_client.fput_object(
            bucket_name=os.environ["RESULTS_BUCKET_NAME"],
            object_name=result_filename,
            file_path=result_path
        )
    except Exception as e:
        raise e
    finally:
        if os.path.exists(source_path):
            os.remove(source_path)
        if os.path.exists(result_path):
            os.remove(result_path)
