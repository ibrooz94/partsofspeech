import nltk
import json
import os

from flask import Flask, render_template, request, flash, redirect, url_for, session
from flask_restful import Resource, Api
from flask_cors import CORS

from nltk.tokenize import word_tokenize

from models import Result, db
from utils import clean_text, get_topics, pos_class
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY")

api = Api(app)
CORS(app)

nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("universal_tagset")

app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{app.root_path}/results_db.sqlite3"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db.init_app(app) 

app.jinja_env.filters['pos_class'] = pos_class

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        text = request.form["text"]
        topics = get_topics(text)
        pos_tags = nltk.pos_tag(topics, tagset="universal")

        session['pos_tags'] = pos_tags

        return render_template(
            "result.html", text=text, pos_tags=pos_tags
        )
    else:
        return render_template("index.html")

@app.route('/save', methods=['POST'])
def save_result():
    text = request.form["text"]
    username = request.form["username"]

    pos_tags_list = session.get('pos_tags', None)

    pos_tags_json = json.dumps(pos_tags_list)
    
    result_obj = Result(text=text, pos_tags=pos_tags_json, username=username)
    db.session.add(result_obj)
    db.session.commit()

    flash("Saved")

    return redirect(url_for('index'))

@app.route('/results')
def list_results():
    username = request.args.get('username')
    start_date = request.args.get('from-timestamp')
    end_date = request.args.get('to-timestamp')

    query = Result.query

    if username:
        query = query.filter_by(username=username)

    if start_date and end_date:
        query = query.filter(Result.timestamp.between(start_date, end_date))
    elif start_date:
        query = query.filter(Result.timestamp >= start_date)
    elif end_date:
        query = query.filter(Result.timestamp <= end_date)

    saved_results = query.all()

    return render_template('saved.html', saved_results=saved_results)

@app.route('/results/<int:result_id>')
def results(result_id):
    result = Result.query.get_or_404(result_id)
    pos_tags = json.loads(result.pos_tags)
    return render_template('result.html', text = result.text, pos_tags=pos_tags, saved=True)

@app.route("/partspeech", methods=["GET", "POST"])
def partspeech():
    if request.method == "POST":
        text = request.form["text"]
        pos_tags = nltk.pos_tag(word_tokenize(clean_text(text)), tagset="universal")
        return render_template("result.html", text=text, pos_tags=pos_tags)
    else:
        return render_template("partspeech.html")


class PartsOfSpeech(Resource):
    def post(self):
        data = request.get_json(force=True)
        text = data["text"]
        tokens = word_tokenize(clean_text(text))
        parts_of_speech = nltk.pos_tag(tokens, tagset="universal")
        return {"input": text, "parts_of_speech": parts_of_speech}

api.add_resource(PartsOfSpeech, "/pos")

with app.app_context():
    db.create_all()

if __name__ == "__main__":
    app.run(debug=True)
