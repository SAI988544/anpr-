import os
import logging
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from models import db, User, Vehicle, VehicleLog  # Import db from models.py


logging.getLogger('matplotlib').setLevel(logging.ERROR)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Create the app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "your-very-secret-key")

# Configure the database
app.config["SQLALCHEMY_DATABASE_URI"] = "mysql+pymysql://root:Kalla%40143@localhost:3306/anpr_db"
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Initialize the database with the app
db.init_app(app)

# Set up the login manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'auth.login'

# Configure file upload settings
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'static/uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Register blueprints
from routes.auth import auth_bp
from routes.admin import admin_bp
from routes.security import security_bp
from routes.dashboard import dashboard_bp
from routes.__init__ import main_bp

app.register_blueprint(main_bp) 
app.register_blueprint(auth_bp)
app.register_blueprint(admin_bp)
app.register_blueprint(security_bp)
app.register_blueprint(dashboard_bp)

# Load user from user_id
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Create tables inside the app context
with app.app_context():
    db.create_all()

if __name__ == "__main__":
    app.run(debug=True)
