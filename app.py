

from flask import Flask, render_template, request
from flask_restful import Resource, Api
from flask_cors import CORS
import nltk
import re
from nltk.tokenize import word_tokenize

app = Flask(__name__)
api = Api(app)

CORS(app)

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')


def clean_text(text):
    # Remove any non-alphanumeric characters from the text
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return cleaned_text
class PartsOfSpeech(Resource):
    def post(self):
        data = request.get_json(force=True)
        sentence = data['text'] 
        tokens = word_tokenize(clean_text(sentence))
        parts_of_speech = nltk.pos_tag(tokens, tagset="universal")
        return {'input': sentence,'parts_of_speech': parts_of_speech}

api.add_resource(PartsOfSpeech, '/pos')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        sentence = request.form['sentence']
        pos_tags = nltk.pos_tag(word_tokenize(clean_text(sentence)), tagset="universal")
        return render_template('result.html', sentence=sentence, pos_tags=pos_tags)
    else:
        return render_template('index.html')

@app.template_filter('pos_class')
def pos_class(pos):
    if pos.startswith('V'):
        return 'verb'
    elif pos.startswith('N'):
        return 'noun'
    elif pos.startswith('ADJ'):
        return 'adjective'
    elif pos.startswith('ADV'):
        return 'adverb'
    elif pos.startswith('DET'):
        return 'det'
    elif pos.startswith('PRON'):
        return 'pron'
    elif pos.startswith('CONJ'):
        return 'conj'
    elif pos.startswith('ADP'):
        return 'adp'
    else:
        return ''
    
if __name__ == '__main__':
    app.run(debug=True)
