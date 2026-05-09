from flask import Flask
from app.settings import jwt
import os
from datetime import timedelta
from flask import Blueprint
from app.settings import Base, engine
from app.settings import mail
from app.models.User import User
from app.models.Document import Document
from app.models.Batch import Batch


def create_app():
    app = Flask(__name__)

    Base.metadata.create_all(bind=engine)

    app.config["JWT_SECRET_KEY"] = os.environ["JWT_SECRET_KEY"]
    app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=1)

    jwt.init_app(app)

    app.config['MAIL_SERVER'] = os.environ["MAIL_SERVER"]
    app.config['MAIL_PORT'] = int(os.environ["MAIL_PORT"])
    app.config['MAIL_USE_TLS'] = True if os.environ["MAIL_USE_TLS"] == "True" else False
    app.config['MAIL_USERNAME'] = os.environ["MAIL_USERNAME"]
    app.config['MAIL_PASSWORD'] = os.environ["MAIL_PASSWORD"]
    app.config['MAIL_DEFAULT_SENDER'] = os.environ["MAIL_USERNAME"]

    mail.init_app(app)

    from app.auth.routes import auth_bp
    from app.documents.routes import documents_bp
    from app.key.routes import api_key_bp

    root = Blueprint('root', __name__)

    root.register_blueprint(auth_bp, url_prefix='/auth')
    root.register_blueprint(documents_bp, url_prefix='/documents')
    root.register_blueprint(api_key_bp, url_prefix='/api-key')

    app.register_blueprint(root, url_prefix='/api')

    return app
