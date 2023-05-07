from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Result(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username= db.Column(db.String(64))
    text= db.Column(db.Text)
    pos_tags = db.Column(db.String)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
