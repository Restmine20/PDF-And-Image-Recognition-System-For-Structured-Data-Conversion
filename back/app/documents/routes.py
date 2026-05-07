import os
import uuid
import io
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from app.models.Batch import Batch
from app.models.Document import Document
from app.settings import SessionLocal, minio_client, global_minio_client, jwt
from app.utils import auth_required
from app.worker_stub import process
from celery.result import AsyncResult
from app.worker_stub import celery as celery_app
from flask import g
from datetime import timedelta

documents_bp = Blueprint('documents', __name__)


@documents_bp.route('', methods=['POST'])
@auth_required
def upload_documents():
    current_user_id = g.current_user_id
    files = request.files.getlist('files')

    if not files:
        return jsonify({"error": "No files selected"}), 400

    session = SessionLocal()

    try:
        new_batch = Batch(user_id=current_user_id)

        session.add(new_batch)

        session.commit()

        batch_id = new_batch.id

    except Exception as e:
        session.rollback()
        return jsonify({"error": "Database error"}), 500
    finally:
        session.close()

    for file in files:
        session = SessionLocal()

        _, extension = os.path.splitext(file.filename)

        unique_part = str(uuid.uuid4())
        unique_filename = unique_part + extension
        unique_result_filename = unique_part + ".xlsx"
        content = file.read()

        try:
            minio_client.put_object(
                bucket_name=os.environ["UPLOADS_BUCKET_NAME"],
                object_name=unique_filename,
                data=io.BytesIO(content),
                length=len(content),
                content_type=file.content_type
            )

            task_id = str(uuid.uuid4())

            new_doc = Document(
                original_filename=file.filename,
                filename=unique_filename,
                batch_id=batch_id,
                task_id=task_id,
                result_filename=unique_result_filename
            )

            session.add(new_doc)

            session.commit()

            process.apply_async(args=(unique_filename,), task_id=task_id)

        except Exception as e:
            session.rollback()
            minio_client.remove_object(os.environ["UPLOADS_BUCKET_NAME"], unique_filename)
            return jsonify({"error": "Database error"}), 500
        finally:
            session.close()

    return {"jobId": batch_id}, 200


@documents_bp.route('/<job_id>', methods=['GET'])
def get_job_status(job_id):
    session = SessionLocal()
    try:
        documents = session.query(Document).filter_by(batch_id=job_id).all()

        if not documents:
            return jsonify({"jobId": job_id, "errorMessage": "Job not found"}), 404

        statuses = set()

        for document in documents:
            res = AsyncResult(document.task_id, app=celery_app)
            document.status = res.status
            statuses.add(res.status)

        session.commit()

        if "PENDING" in statuses:
            return jsonify({"jobId": job_id, "status": "PENDING"}), 200
        if "STARTED" in statuses:
            return jsonify({"jobId": job_id, "status": "STARTED"}), 200

        files_list = []
        for document in documents:
            if document.status == "SUCCESS":
                url = global_minio_client.presigned_get_object(
                    bucket_name=os.environ["RESULTS_BUCKET_NAME"],
                    object_name=document.result_filename,
                    expires=timedelta(minutes=5)
                )
                files_list.append({"name": document.original_filename, "resultUrl": url})
        return jsonify({"jobId": job_id, "status": "SUCCESS", "files": files_list}), 200
    except Exception as e:
        session.rollback()
        return jsonify({"jobId": job_id, "errorMessage": "Database error"}), 517
    finally:
        session.close()


@documents_bp.route('/history', methods=['GET'])
@auth_required
def get_history():
    current_user_id = g.current_user_id
    if current_user_id is None:
        return jsonify({"error": "User required"}), 401

    session = SessionLocal()
    all_documents = []
    try:
        batches = session.query(Batch).filter_by(user_id=current_user_id).all()

        for batch in batches:
            documents = session.query(Document).filter_by(batch_id=batch.id).all()
            all_documents.extend(documents)

        if not all_documents:
            return jsonify([]), 200

        response_list = []

        for document in all_documents:
            res = AsyncResult(document.task_id, app=celery_app)
            if res.status != "PENDING":
                document.status = res.status
            url = global_minio_client.presigned_get_object(
                bucket_name=os.environ["RESULTS_BUCKET_NAME"],
                object_name=document.result_filename,
                expires=timedelta(minutes=5)
            )
            response_list.append({"jobId": document.batch_id,
                                  "filename": document.original_filename,
                                  "status": document.status,
                                  "resultUrl": url})

        session.commit()

        return jsonify(response_list), 200
    except Exception as e:
        session.rollback()
        return jsonify({"error": "Database error"}), 500
    finally:
        session.close()
