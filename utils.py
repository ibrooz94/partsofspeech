from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.parsing.preprocessing import STOPWORDS
from gensim.utils import simple_preprocess
import contractions

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
    cleaned_text = " ".join([contractions.fix(word) for word in text.split()])
    return cleaned_text

def pos_class(pos):
    pos_classes = {"V": "verb", "N": "noun", "ADJ": "adjective", "ADV": "adverb", "DET": "det", "PRON": "pron", "CONJ": "conj", "ADP": "adp"}

    for key, value in pos_classes.items():
        if pos.startswith(key):
            return value

    return ""