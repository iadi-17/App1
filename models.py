from flask import redirect , url_for , session
from flask_login import UserMixin
from sqlalchemy.orm import backref
from __init__ import db 

#from main import app


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True) # primary keys are required by SQLAlchemy
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))
    name = db.Column(db.String(1000))
    #posts = db.relationship('Blogpost' , backref = id,lazy = True)

class Blogpost(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(50))
    subtitle = db.Column(db.String(50))
    author = db.Column(db.String(20))
    date_posted = db.Column(db.DateTime)
    content = db.Column(db.Text)
   
class FileContents(db.Model):
    id = db.Column(db.Integer , primary_key = True)
    name= db.Column(db.String(1000))
    data = db.Column(db.LargeBinary)

class City(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)

