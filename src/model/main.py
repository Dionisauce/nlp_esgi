from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC 
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from joblib import dump, load 

def make_model(isdumpable=True):
    model = Pipeline([
        ("count_vectorizer", CountVectorizer()),
        #("tfidf_vectorizer", TfidfVectorizer()),

        # Régression logistique (avec accent): 93.28% (Lemmatization)
        # tfidf_vectorizer : 87.98%
        #("regression_logistique", LogisticRegression()) 

        # Random Forest (avec accent): 93.85% (Lemmatization)
        # tfidf_vectorizer : 93.84%
        #("random_forest", RandomForestClassifier()),

        # Support Vector Machine (SVM) (avec accent et sans accent): 93.27%
        # tfidf_vectorizer : 91.99%
        #("SVM", SVC()),

        # Naive Bayes (sans accent): 94.13% (Lemmatization)
        # tfidf_vectorizer : 85.83%
        ("Naive_Bayes", MultinomialNB()),
    ])
    if isdumpable:
        return DumpableModel(model)
    else:
        return model
    

class DumpableModel:
    def __init__(self, model):
        self.model = model
    
    def fit(self, X, y=None):
        return self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def dump(self, filename_output):
        dump(self.model, filename_output)

    def load(self, filename_input):
        self.model = load(filename_input)
