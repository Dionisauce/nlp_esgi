import re
import unidecode
import nltk 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import FrenchStemmer


# Téléchargement des banques de mots
#nltk.download('stopwords')
#nltk.download('wordnet')

stop_words = set(stopwords.words('french'))
lemmatizer = WordNetLemmatizer()
stemmer = FrenchStemmer()

def traitement_txt(text):
    try:
        # On vérifie si text est de type 'str'
        if isinstance(text, str):
            # Tout en minuscule
            #text = text.lower()

            # On retire les accents
            text = unidecode.unidecode(text)

            # On enlève les stopwords 
            text = ' '.join(word for word in text.split() if word not in stop_words)

            # Lemmatisation : on uniformise les mots
            text = ' '.join(lemmatizer.lemmatize(word) for word in text.split())

            # Stemming : on tronque les mots pour conserver le radical
            #text = ' '.join(stemmer.stem(word) for word in text.split())

            return text
        else:
            print("Input pas au format string")
    except Exception as e:
        print(f"Une erreur s'est produite: {str(e)}")

def make_features(df, task):
    y = get_output(df, task)

    X = df["video_name"].apply(traitement_txt)

    print(X[4]) # print données spécifiques

    return X, y


def get_output(df, task):
    if task == "is_comic_video":
        y = df["is_comic"]
    elif task == "is_name":
        y = df["is_name"]
    elif task == "find_comic_name":
        y = df["comic_name"]
    else:
        raise ValueError("Unknown task")
    
    return y
