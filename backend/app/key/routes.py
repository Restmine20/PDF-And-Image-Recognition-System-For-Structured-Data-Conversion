from flask import Blueprint, jsonify
from app.models.User import User
from flask_jwt_extended import jwt_required, get_jwt_identity
from app.settings import SessionLocal, minio_client, jwt
import secrets


api_key_bp = Blueprint('api-key', __name__)


@api_key_bp.route('', methods=['POST'])
@jwt_required()
def generate_api_key():
    current_user_id = get_jwt_identity()

    session = SessionLocal()
    try:
        user = session.query(User).filter(User.id == current_user_id).first()
        if not user:
            return jsonify({"error": "User does not exist"}), 404

        api_key = secrets.token_hex(32)
        user.api_key = api_key

        session.commit()

        return jsonify({"key": api_key}), 200
    except Exception as e:
        session.rollback()
        return jsonify({"error": "Server error"}), 500
    finally:
        session.close()
