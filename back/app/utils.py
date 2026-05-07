from flask_jwt_extended import verify_jwt_in_request, get_jwt_identity, get_jwt
from functools import wraps
from flask import request, g
from app.models.User import User
from app.settings import SessionLocal, jwt_redis_blocklist


def auth_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):

        session = SessionLocal()
        try:
            user_id = None

            try:
                verify_jwt_in_request()

                jti = get_jwt()["jti"]

                if not jwt_redis_blocklist.exists(jti):
                    user_id = get_jwt_identity()
            except Exception as e:
                pass

            if not user_id:
                api_key = request.headers.get('X-API-Key')
                if api_key is not None:
                    user = session.query(User).filter(User.api_key == api_key).first()
                    if user:
                        user_id = user.id
            g.current_user_id = user_id
            return f(*args, **kwargs)
        finally:
            session.close()
    return decorated
