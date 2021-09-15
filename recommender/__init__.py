from flask import Flask,make_response
from recommender.apicontroller import apicontroller
import recommender

app = Flask(__name__)
#register routes from apicontroller.py
app.register_blueprint(apicontroller)

