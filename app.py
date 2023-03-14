

from flask import Flask, render_template, request
from flask_restful import Resource, Api
from flask_cors import CORS
import nltk
import contractions
from nltk.tokenize import word_tokenize

from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.parsing.preprocessing import STOPWORDS
from gensim.utils import simple_preprocess

app = Flask(__name__)
api = Api(app)

CORS(app)

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')


def get_topics(text):
    # Preprocess the text by tokenizing and removing stop words
    stop_words = STOPWORDS
    tokens = [token for token in simple_preprocess(text) if token not in stop_words]
    
    # Create a dictionary from the preprocessed text
    dictionary = corpora.Dictionary([tokens])
    
    # Convert the text to a bag-of-words format
    bow_corpus = [dictionary.doc2bow(tokens)]
    
    # Train an LDA model on the bag-of-words corpus
    lda_model = LdaModel(bow_corpus, num_topics=10, id2word=dictionary)
    
    # Get the top topics and their scores for the text
    topics = lda_model.show_topic(0)
    
    # Return just the topic strings
    return [topic[0] for topic in topics]

def clean_text(text):
    cleaned_text = ' '.join([contractions.fix(word) for word in text.split()])
    return cleaned_text


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        sentence = request.form['sentence']
        text = get_topics(sentence)
        pos_tags = nltk.pos_tag(text, tagset="universal")
        return render_template('result.html', sentence=sentence, pos_tags=pos_tags, topics = text)
    else:
        return render_template('index.html')
    
@app.route('/partspeech', methods=['GET', 'POST'])
def topics():
    if request.method == 'POST':
        sentence = request.form['sentence']
        pos_tags = nltk.pos_tag(word_tokenize(clean_text(sentence)), tagset="universal")
        return render_template('result.html', sentence=sentence, pos_tags=pos_tags)
    else:
        return render_template('partspeech.html')

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
class PartsOfSpeech(Resource):
    def post(self):
        data = request.get_json(force=True)
        sentence = data['text'] 
        tokens = word_tokenize(clean_text(sentence))
        parts_of_speech = nltk.pos_tag(tokens, tagset="universal")
        return {'input': sentence,'parts_of_speech': parts_of_speech}

api.add_resource(PartsOfSpeech, '/pos')
    
if __name__ == '__main__':
    app.run(debug=True)
