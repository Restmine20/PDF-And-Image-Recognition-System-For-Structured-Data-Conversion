from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity, get_jwt
from app.settings import SessionLocal, jwt_redis_blocklist
from app.models.User import User
import secrets
from flask_mail import Message
from app.settings import mail


auth_bp = Blueprint('auth', __name__)


@auth_bp.route('/register', methods=['POST'])
def register():
    data = request.get_json()

    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    session = SessionLocal()

    try:
        if session.query(User).filter(User.email == email).first():
            return jsonify({"error": "User already exists"}), 409

        new_user = User(email=email)
        new_user.set_password(password)

        session.add(new_user)
        session.commit()

        access_token = create_access_token(identity=new_user.id)

        message = Message(
            subject="Успешная регистрация",
            recipients=[email],
            body="Вы успешно зарегистрировались на сайте"
        )

        mail.send(message)

        return jsonify({"token": access_token, "user": {"email": email}}), 201
    except Exception as e:
        session.rollback()
        return jsonify({"error": "Database error"}), 500
    finally:
        session.close()


@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json()

    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    session = SessionLocal()

    try:
        user = session.query(User).filter(User.email == email).first()

        if not user or not user.check_password(password):
            return jsonify({"error": "Invalid credentials"}), 401

        access_token = create_access_token(identity=user.id)

        return jsonify({"token": access_token, "user": {"email": email}}), 200
    except Exception as e:
        return jsonify({"error": "Login error"}), 500

    finally:
        session.close()


@auth_bp.route('/me', methods=['GET'])
@jwt_required()
def me():
    current_user_id = get_jwt_identity()
    session = SessionLocal()
    try:
        user = session.query(User).filter(User.id == current_user_id).first()
        return jsonify({"user": {"email": user.email}}), 200
    except Exception as e:
        return jsonify({"error": "User not found"}), 401
    finally:
        session.close()


@auth_bp.route('/logout', methods=['POST'])
@jwt_required()
def logout():
    jwt_redis_blocklist.set(get_jwt()["jti"], "revoked", ex=3600)

    return "", 204


@auth_bp.route('/password/reset', methods=['POST'])
def reset_password():
    data = request.get_json()

    email = data.get('email')

    session = SessionLocal()
    try:
        user = session.query(User).filter(User.email == email).first()

        new_password = secrets.token_hex(16)

        message = Message(
            subject="Восстановление пароля",
            recipients=[email],
            body=f"Новый пароль от аккаунта: {new_password}"
        )

        user.set_password(new_password)

        session.commit()

        mail.send(message)
    except Exception as e:
        session.rollback()
    finally:
        session.close()
    return "", 204


@auth_bp.route('/password/change', methods=['POST'])
@jwt_required()
def change_password():
    data = request.get_json()

    current_user_id = get_jwt_identity()
    new_password = data.get("newPassword")

    session = SessionLocal()
    try:
        user = session.query(User).filter(User.id == current_user_id).first()

        user.set_password(new_password)

        message = Message(
            subject="Смена пароля",
            recipients=[user.email],
            body=f"Пароль вашего аккаунта был изменен. Новый пароль: {new_password}"
        )

        user.set_password(new_password)

        session.commit()

        mail.send(message)
    except Exception as e:
        session.rollback()
    finally:
        session.close()
    return "", 204
