from flask import Flask

PATH = 'D:/Freelance big/Mohammed A/'

# creates the flask app to run as web interface
app = Flask(__name__, template_folder=PATH + 'templates/')
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = PATH + 'TEST/'