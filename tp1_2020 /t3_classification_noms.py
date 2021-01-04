# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import glob
import os
import string
import unicodedata
import json

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

datafiles = "./data/names/*.txt"  # les fichiers pour construire vos modèles
test_filename = './data/test-names-t3.txt'  # le fichier contenant les données de test pour évaluer vos modèles

names_by_origin = {}  # un dictionnaire qui contient une liste de noms pour chaque langue d'origine
all_origins = []  # la liste des 18 langues d'origines de noms 

# Fonctions utilitaires pour lire les données d'entraînement et de test - NE PAS MODIFIER

def load_names():
    """Lecture des noms et langues d'origine d'un fichier. Par la suite,
       sauvegarde des noms pour chaque origine dans le dictionnaire names_by_origin."""
    for filename in find_files(datafiles):
        origin = get_origin_from_filename(filename)
        all_origins.append(origin)
        names = read_names(filename)
        names_by_origin[origin] = names
        

def find_files(path):
    """Retourne le nom des fichiers contenus dans un répertoire.
       glob fait le matching du nom de fichier avec un pattern - par ex. *.txt"""
    return glob.glob(path)


def get_origin_from_filename(filename):
    """Passe-passe qui retourne la langue d'origine d'un nom de fichier.
       Par ex. cette fonction retourne Arabic pour "./data/names/Arabic.txt". """
    return os.path.splitext(os.path.basename(filename))[0]


def read_names(filename):
    """Retourne une liste de tous les noms contenus dans un fichier."""
    with open(filename, encoding='utf-8') as f:
        names = f.read().strip().split('\n')
    return [unicode_to_ascii(name) for name in names]


def unicode_to_ascii(s):
    """Convertion des caractères spéciaux en ascii. Par exemple, Hélène devient Helene.
       Tiré d'un exemple de Pytorch. """
    all_letters = string.ascii_letters + " .,;'"
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

def load_test_names(filename):
    """Retourne un dictionnaire contenant les données à utiliser pour évaluer vos modèles.
       Le dictionnaire contient une liste de noms (valeurs) et leur origine (clé)."""
    with open(filename, 'r') as fp:
        test_data = json.load(fp)
    return test_data

#---------------------------------------------------------------------------
# Fonctions à développer pour ce travail - Ne pas modifier les signatures et les valeurs de retour
unigramme_bayes_classifier = MultinomialNB()
bigramme_bayes_classifier = MultinomialNB()
trigramme_bayes_classifier = MultinomialNB()
multigramme_bayes_classifier = MultinomialNB()

Tfidf_unigramme_bayes_classifier = MultinomialNB()
Tfidf_bigramme_bayes_classifier = MultinomialNB()
Tfidf_trigramme_bayes_classifier = MultinomialNB()
Tfidf_multigramme_bayes_classifier = MultinomialNB()

unigramme_logit_classifier = LogisticRegression(max_iter=400)
bigramme_logit_classifier = LogisticRegression(max_iter=200)
trigramme_logit_classifier = LogisticRegression(max_iter=200)
multigramme_logit_classifier = LogisticRegression(max_iter=400)

Tfidf_unigramme_logit_classifier = LogisticRegression(max_iter=400)
Tfidf_bigramme_logit_classifier = LogisticRegression(max_iter=200)
Tfidf_trigramme_logit_classifier = LogisticRegression(max_iter=200)
Tfidf_multigramme_logit_classifier = LogisticRegression(max_iter=400)

unigramme_bayes_vectorizer = CountVectorizer(lowercase=True, analyzer='char', ngram_range=(1, 1))
bigramme_bayes_vectorizer = CountVectorizer(lowercase=True, analyzer='char', ngram_range=(2, 2))
trigramme_bayes_vectorizer = CountVectorizer(lowercase=True, analyzer='char', ngram_range=(3, 3))
multigramme_bayes_vectorizer = CountVectorizer(lowercase=True, analyzer='char', ngram_range=(1, 3))

Tfidf_unigramme_bayes_vectorizer = TfidfVectorizer(lowercase=True, analyzer='char', ngram_range=(1, 1))
Tfidf_bigramme_bayes_vectorizer = TfidfVectorizer(lowercase=True, analyzer='char', ngram_range=(2, 2))
Tfidf_trigramme_bayes_vectorizer = TfidfVectorizer(lowercase=True, analyzer='char', ngram_range=(3, 3))
Tfidf_multigramme_bayes_vectorizer = TfidfVectorizer(lowercase=True, analyzer='char', ngram_range=(1, 3))

def train_classifiers():
    load_names()
    # Vous ajoutez à partir d'ici tout le code dont vous avez besoin
    # pour construire les différentes versions de classificateurs de langues d'origines.
    # Voir les consignes de l'énoncé du travail pratique pour déterminer les différents modèles à entraîner.
    #
    # On suppose que les données d'entraînement ont été lues (load_names) et sont disponibles (names_by_origin).
    #
    # Vous pouvez ajouter au fichier toutes les fonctions que vous jugerez nécessaire.
    # Merci de ne pas modifier les signatures (noms de fonctions et arguments) déjà présentes dans le fichier.
    #
    # Votre code à partir d'ici...
    #
    X_train = list()
    y_train = list()
    for key in names_by_origin.keys():
        value = names_by_origin[key]
        X_train = X_train + value
        for i in range(len(value)):
            y_train.append(key)

    X_train = [unicode_to_ascii(s.strip()) for s in X_train]
    y_train = [unicode_to_ascii(s.strip()) for s in y_train]

    print("\n")
    print("=====================================================")
    print("                comptes de N-grammes                 ")
    print("=====================================================")
    print("\n")

    
    print("=========== training models unigrame ================")
    unigramme_bayes_vectorizer.fit(X_train)
    X_train_vectorized = unigramme_bayes_vectorizer.transform(X_train)
    unigramme_bayes_classifier.fit(X_train_vectorized, y_train)
    print("unigramme_bayes_classifier : terminer")
    unigramme_logit_classifier.fit(X_train_vectorized, y_train)
    print("unigramme_logit_classifier : terminer")

    print("============ training models bigrame ================")
    bigramme_bayes_vectorizer.fit(X_train)
    X_train_vectorized = bigramme_bayes_vectorizer.transform(X_train)
    bigramme_bayes_classifier.fit(X_train_vectorized, y_train)
    print("bigramme_bayes_classifier : terminer")
    bigramme_logit_classifier.fit(X_train_vectorized, y_train)
    print("bigramme_logit_classifier : terminer")

    print("=========== training models trigrame ================")
    trigramme_bayes_vectorizer.fit(X_train)
    X_train_vectorized = trigramme_bayes_vectorizer.transform(X_train)
    trigramme_bayes_classifier.fit(X_train_vectorized, y_train)
    print("trigramme_bayes_classifier : terminer")
    trigramme_logit_classifier.fit(X_train_vectorized, y_train)
    print("trigramme_logit_classifier : terminer")
    
    print("=========== training models multigrame ===============")
    multigramme_bayes_vectorizer.fit(X_train)
    X_train_vectorized = multigramme_bayes_vectorizer.transform(X_train)
    multigramme_bayes_classifier.fit(X_train_vectorized, y_train)
    print("multigramme_bayes_classifier : terminer")
    multigramme_logit_classifier.fit(X_train_vectorized, y_train)
    print("multigramme_logit_classifier : terminer")


    print("\n")
    print("=====================================================")
    print("                     poids tfidf                     ")
    print("=====================================================")
    print("\n")

    
    print("=========== training models unigrame ================")
    Tfidf_unigramme_bayes_vectorizer.fit(X_train)
    X_train_vectorized = Tfidf_unigramme_bayes_vectorizer.transform(X_train)
    Tfidf_unigramme_bayes_classifier.fit(X_train_vectorized, y_train)
    print("Tfidf_unigramme_bayes_classifier : terminer")
    Tfidf_unigramme_logit_classifier.fit(X_train_vectorized, y_train)
    print("Tfidf_unigramme_logit_classifier : terminer")

    print("============ training models bigrame ================")
    Tfidf_bigramme_bayes_vectorizer.fit(X_train)
    X_train_vectorized = Tfidf_bigramme_bayes_vectorizer.transform(X_train)
    Tfidf_bigramme_bayes_classifier.fit(X_train_vectorized, y_train)
    print("Tfidf_bigramme_bayes_classifier : terminer")
    Tfidf_bigramme_logit_classifier.fit(X_train_vectorized, y_train)
    print("Tfidf_bigramme_logit_classifier : terminer")

    print("=========== training models trigrame ================")
    Tfidf_trigramme_bayes_vectorizer.fit(X_train)
    X_train_vectorized = Tfidf_trigramme_bayes_vectorizer.transform(X_train)
    Tfidf_trigramme_bayes_classifier.fit(X_train_vectorized, y_train)
    print("Tfidf_trigramme_bayes_classifier : terminer")
    Tfidf_trigramme_logit_classifier.fit(X_train_vectorized, y_train)
    print("Tfidf_trigramme_logit_classifier : terminer")
    
    print("=========== training models multigrame ===============")
    Tfidf_multigramme_bayes_vectorizer.fit(X_train)
    X_train_vectorized = Tfidf_multigramme_bayes_vectorizer.transform(X_train)
    Tfidf_multigramme_bayes_classifier.fit(X_train_vectorized, y_train)
    print("Tfidf_multigramme_bayes_classifier : terminer")
    Tfidf_multigramme_logit_classifier.fit(X_train_vectorized, y_train)
    print("Tfidf_multigramme_logit_classifier : terminer")  
    
def get_classifier(type, n=3, weight='tf'):
    # Retourne le classificateur correspondant. On peut appeler cette fonction
    # après que les modèles aient été entraînés avec la fonction train_classifiers
    #
    classifier = trigramme_bayes_classifier

    if weight=='tf' :
        if n == 1 :
            if type == 'naive_bayes' :
                classifier = unigramme_bayes_classifier
            if type == 'logistic_regresion' :
                classifier = unigramme_logit_classifier
        if n == 2 :
            if type == 'naive_bayes' :
                classifier = bigramme_bayes_classifier
            if type == 'logistic_regresion' :
                classifier = bigramme_logit_classifier
        if n == 3 :
            if type == 'naive_bayes' :
                classifier = trigramme_bayes_classifier
            if type == 'logistic_regresion' :
                classifier = trigramme_logit_classifier
        if n == 'multi' :
            if type == 'naive_bayes' :
                classifier = multigramme_bayes_classifier
            if type == 'logistic_regresion' :
                classifier = multigramme_logit_classifier

    if weight=='tfidf' :
        if n == 1 :
            if type == 'naive_bayes' :
                classifier = Tfidf_unigramme_bayes_classifier
            if type == 'logistic_regresion' :
                classifier = Tfidf_unigramme_logit_classifier
        if n == 2 :
            if type == 'naive_bayes' :
                classifier = Tfidf_bigramme_bayes_classifier
            if type == 'logistic_regresion' :
                classifier = Tfidf_bigramme_logit_classifier
        if n == 3 :
            if type == 'naive_bayes' :
                classifier = Tfidf_trigramme_bayes_classifier
            if type == 'logistic_regresion' :
                classifier = Tfidf_trigramme_logit_classifier
        if n == 'multi' :
            if type == 'naive_bayes' :
                classifier = Tfidf_multigramme_bayes_classifier
            if type == 'logistic_regresion' :
                classifier = Tfidf_multigramme_logit_classifier

    return classifier

def get_vectorizer(n=3, weight='tf'):
    # Retourne le classificateur correspondant. On peut appeler cette fonction
    # après que les modèles aient été entraînés avec la fonction train_classifiers
    #

    vectorizer = trigramme_bayes_vectorizer

    if weight=='tf' :
        if n == 1 :
            vectorizer = unigramme_bayes_vectorizer
        if n == 2 :
            vectorizer = bigramme_bayes_vectorizer
        if n == 3 :
            vectorizer = trigramme_bayes_vectorizer
        if n == 'multi' :
            vectorizer = multigramme_bayes_vectorizer

    if weight=='tfidf' :
        if n == 1 :
            vectorizer = Tfidf_unigramme_bayes_vectorizer
        if n == 2 :
            vectorizer = Tfidf_bigramme_bayes_vectorizer
        if n == 3 :
            vectorizer = Tfidf_trigramme_bayes_vectorizer
        if n == 'multi' :
            vectorizer = Tfidf_multigramme_bayes_vectorizer

    return vectorizer
    
def origin(name, type, n=3, weight='tf'):
    # Retourne la langue d'origine prédite pour le nom.
    #   - name = le nom qu'on veut classifier
    #   - type = 'naive_bayes' ou 'logistic_regresion'
    #   - n désigne la longueur des N-grammes. Choix possible = 1, 2, 3, 'multi'
    #   - weight désigne le poids des attributs. Soit tf (comptes) ou tfidf.
    #
    # Votre code à partir d'ici...
    # À compléter...
    #

    classifier = get_classifier(type=type, n=n, weight=weight)
    vectorizer = get_vectorizer(n=n, weight=weight)

    name = unicode_to_ascii(name)
    name = vectorizer.transform([name])
    name_origin = classifier.predict(name)
    
    return name_origin 
    
    
def test_classifier(test_fn, type, n=3, weight='tf'):
    test_data = load_test_names(test_fn)
    print("Nb de noms de test:", len(test_data))

    # Insérer ici votre code pour la classification des questions.
    # Votre code...

    X_test = list()
    y_test = list()
    for key in test_data.keys():
        value = test_data[key]
        X_test = X_test + value
        for i in range(len(value)):
            y_test.append(key)

    X_test_1 = X_test
    X_test = [unicode_to_ascii(s.strip()) for s in X_test]
    y_test = [unicode_to_ascii(s.strip()) for s in y_test]

    classifier = get_classifier(type=type, n=n, weight=weight)
    vectorizer = get_vectorizer(n=n, weight=weight)

    X_test = vectorizer.transform(X_test)
    y_test_pred = classifier.predict(X_test)

    test_accuracy = accuracy_score(y_test, y_test_pred)

    """
    for i in range(len(y_test)) :
        if y_test[i] != y_test_pred[i] :
            print("name : {}   reel : {}   pred : {}".format(X_test_1[i], y_test[i], y_test_pred[i]))

    cm = confusion_matrix(y_test, y_test_pred)
    classes = classifier.classes_
    display_confusion_matrix(cm, classes)
    """
    
    return test_accuracy

def train_classifier(test_fn, type, n=3, weight='tf'):

    X_train = list()
    y_train = list()
    for key in names_by_origin.keys():
        value = names_by_origin[key]
        X_train = X_train + value
        for i in range(len(value)):
            y_train.append(key)

    X_train = [unicode_to_ascii(s.strip()) for s in X_train]
    y_train = [unicode_to_ascii(s.strip()) for s in y_train]

    classifier = get_classifier(type=type, n=n, weight=weight)
    vectorizer = get_vectorizer(n=n, weight=weight)

    X_train = vectorizer.transform(X_train)
    y_train_pred = classifier.predict(X_train)

    train_accuracy = accuracy_score(y_train, y_train_pred)

    return train_accuracy

def display_confusion_matrix(confusion_matrix, classes):
    print("\n\nVersion graphique de la matrice de confusion") 
    df_cm = pd.DataFrame(confusion_matrix, index=classes, columns=classes)
    f, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(df_cm, annot=True, fmt="d", linewidths=.5, ax=ax)
    plt.ylabel('Vrai étiquette ')
    plt.xlabel('Étiquette prédite')
    plt.savefig('matrix_confusion.gif')


def evaluate_classifiers(filename):
    """Fonction utilitaire pour évaluer vos modèles. Aucune contrainte particulière.
       Nous n'utiliserons pas cette fonction pour l'évaluation de votre travail.
       Vous pouvez modifier le nom ou les arguments.
       """
    #test_data = load_test_names(filename)
    types =['naive_bayes', 'logistic_regresion']
    ngrames = [1,2,3,'multi']
    weights = ['tf','tfidf']

    print("\n")

    for weight in weights :
        for type in types :
            for ngrame in ngrames :
                    print("type : {}   n : {}   weight : {}".format(type, ngrame, weight))
                    test_accuracy = test_classifier(filename, type=type, n=ngrame, weight=weight)
                    train_accuracy = train_classifier(filename, type=type, n=ngrame, weight=weight)
                    print("train_accuracy : ", train_accuracy)
                    print("test_accuracy : ", test_accuracy)
                    print("==================================================================================")


if __name__ == '__main__':
    # Vous pouvez modifier cette section comme bon vous semble
    load_names()
    """
    print("Les {} langues d'origine sont: \n{}".format(len(all_origins), all_origins))
    chinese_names = names_by_origin["Chinese"]
    print("\nQuelques noms chinois : \n", chinese_names[:20])
    """

    train_classifiers()
    some_name = "bouyahia"

    classifier = get_classifier('logistic_regresion', n=3, weight='tf')
    print("\nType de classificateur: ", classifier)

    some_origin = origin(some_name, 'naive_bayes', n='multi', weight='tfidf')
    #print("\nLangue d'origine de {}: {}".format(some_name, some_origin))

    #test_accuracy = test_classifier(test_filename, type='logistic_regresion', n=3, weight='tf')

    #print("test_accuracy : ", test_accuracy)

    #evaluate_classifiers(test_filename)

    test_accuracy = test_classifier(test_filename, 'logistic_regresion', n='multi', weight='tf')
    
    """
    test_names = load_test_names(test_filename)
    print("\nLes données pour tester vos modèles sont:")
    for org, name_list in test_names.items():
        print("\t{} : {}".format(org, name_list))
    evaluate_classifiers(test_filename)
    """

