import pandas as pd

def make_dataset(filename):
    try:
        data = pd.read_csv(filename)
        return data
    except FileNotFoundError:
        print(f"Pas de fichier dans le chemin suivant : {filename}")
        return None
    except pd.errors.EmptyDataError:
        print(f"Fichier vide : {filename}")
        return None
    except pd.errors.ParserError:
        print(f"Erreur lors de la lecture du fichier : {filename}")
        return None
    


