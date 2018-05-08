from flask import Flask, render_template, request
#from config import Config
import os
import pickle
import nltk
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
mystops = set(stopwords.words("english"))

from bs4 import BeautifulSoup
import re
# Transformation de la feature 'Body'
def body_to_words(value):
    # 1. Remove HTML
    review_text = BeautifulSoup(value, "lxml").get_text() 
    # 2. Remove special characters
    letters_only = re.sub("[^a-zA-Z0-9#]", " ", review_text) 
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    # 4. Remove stop words
    meaningful_words = [w for w in words if not w in mystops]
    # 5. Lemme
    porter = nltk.PorterStemmer()
    meaningful_words = [porter.stem(w) for w in meaningful_words]
    # 6. Join the words back into one string separated by space, and return the result.
    return( " ".join( meaningful_words ))   
    
# Méthode pour récupérer les meilleurs tags en fonction du pourcentage de la prédiction
def get_best_tags(clf, X, lb, n_tags=5):
    try:
        decfun = []
        if hasattr(clf, 'decision_function'):
            decfun = clf.decision_function(X)
        elif hasattr(clf, 'predict_proba'):
            decfun = clf.predict_proba(X)
        else:
            return None
    
        best_tags = np.argsort(decfun)[:, :-(n_tags+1): -1]
    
        return lb.classes_[best_tags]
    except Exception as e:
        return str(e)
    
app = Flask(__name__)

app.config.from_object('config')
CT_DIR = app.config['BASESAVE']

def load_obj(name):
    with open(os.path.join(CT_DIR, name + '.pkl'), 'rb') as f:
        return pickle.load(f)

# Modèle
model_Classifier = load_obj('ClassifierSVC')
model_Binarizer = load_obj('MultiLabelBinarizer')

CT_TITRE = "title"
CT_QUESTION = "question"

@app.route('/')
def index():
    # Affichage de la page d'accueil avec les aéroports et les compagnies
    return render_template('index.html')

@app.route('/tag/', methods=['POST'])
def hello():
    if request.method == 'POST':
        try:
            df = pd.DataFrame(columns=['Body', 'Title'])
            df.loc[0] = 0

            # Récupération des paramètres envoyés par la requête
            myTitle = request.values[CT_TITRE]
            myQuestion = request.values[CT_QUESTION]

            df['Body'][0] = myQuestion
            df['Title'][0] = myTitle
            df['TextCleaned'] = df['Title'].apply(body_to_words) + ' ' + df['Body'].apply(body_to_words)
            ret = get_best_tags(model_Classifier, df['TextCleaned'], model_Binarizer)

            return ' '.join(t for t in ret[0])
        except Exception as e:
            return str(e)
    return "NO POST"

