
from flask import Flask, render_template, request
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        sentence = request.form['sentence']
        pos_tags = nltk.pos_tag(nltk.word_tokenize(sentence), tagset="universal")
        return render_template('result.html', pos_tags=pos_tags)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
