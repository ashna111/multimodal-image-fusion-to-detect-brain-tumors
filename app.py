import os
from datetime import datetime
from flask import Flask, render_template, request, url_for

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/upload", methods=['POST'])
def upload():
    target = os.path.join(APP_ROOT, 'static/')
    if not os.path.isdir(target):
        os.mkdir(target)

    mri_file=request.files['mri']
    ct_file=request.files['ct']

    destination1 = "/".join([target, "mri.jpg"])
    mri_file.save(destination1)

    destination2 = "/".join([target, "ct.jpg"])
    ct_file.save(destination2)
    
    return render_template("registration.html")

if __name__ == "__main__":
    app.run(debug=True)