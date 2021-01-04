# -*- coding: utf-8 -*-
import json
import spacy
import nltk
import pandas as pd
import numpy as np
import math
import negation_conversion as nc
nltk.download('sentiwordnet')
nlp = spacy.load('en_core_web_sm')
from nltk.corpus import sentiwordnet as swn 
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
import scipy.sparse as sp
import json
from sklearn.utils import shuffle

reviews_dataset = {
    'train_pos_fn' : "./data/train_positive.txt",
    'train_neg_fn' : "./data/train_negative.txt",
    'test_pos_fn' : "./data/test_positive.txt",
    'test_neg_fn' : "./data/test_negative.txt"
}

def get_Jurafsky_X_y(X, y, data, is_pos=1):

    tagged_map = {
        "VERB" : "v",
        "NOUN" : "n",
        "ADJ"  : "a",
        "ADV"  : "r",
    }
    tagged_list = ["VERB", "NOUN", "ADJ", "ADV"]
    negation_words = ["no", "hardly", "barely", "none", "nothing", "without","n't","nobody","never","neither","nowhere","scarcely", "seldom"]
    i=0
    
    for text in data :
        number_word_pos = 0
        number_word_neg = 0
        presence_word_of_neg = 0
        compte_pronon_1_2 = 0
        nomber_exclamation_mark = 0
        len_text = 0 
        
        doc = nlp(text)
        word_list = []
        pos_tag_list = []
        token_list = []
        features = []

        for token in doc :
            pos_tag_list.append(token.pos_)
            word_list.append(token.text)
            token_list.append(token)      

        for i in range(len(word_list)) :
            if pos_tag_list[i] in tagged_list :               
                list_score = list(swn.senti_synsets(word_list[i], tagged_map[pos_tag_list[i]]))
                if len(list_score) > 0 : 
                    list_score = [list_score[0].pos_score(), list_score[0].neg_score(), list_score[0].obj_score()]
                    index_max = list_score.index(max(list_score))
                    if index_max == 0 :
                        number_word_pos += 1
                    if index_max == 1 :
                        number_word_neg += 1        

        intersection_list = list(set(negation_words) & set(word_list))
        
        compte_pronon_1_2 = word_list.count('i') + word_list.count('I') + word_list.count('you')+ word_list.count('we') + word_list.count('me')
        nomber_exclamation_mark = word_list.count('!')
        len_text = math.log(len(set(word_list)))
        if len(intersection_list) > 0 :
            presence_word_of_neg = 1

        features = [
            number_word_pos,
            number_word_neg,
            presence_word_of_neg,
            compte_pronon_1_2,
            nomber_exclamation_mark,
            len_text
        ]
        X.append(features)
        y.append(is_pos)

    return X, y

def get_Ohana_X_y(X, y, data, is_pos=1):

    tagged_map = {
        "VERB" : "v",
        "NOUN" : "n",
        "ADJ"  : "a",
        "ADV"  : "r",
    }
    tagged_list = [
        "VERB",
        "NOUN",
        "ADJ",
        "ADV"
    ]

    for text in data :
        number_nom = 0
        number_nom_proper = 0
        number_adj = 0
        number_verb = 0
        number_adv = 0
        number_interjection = 0
        number_sentence = 0
        avg_len_sentence = 0.
        number_useful_word = 0
        cumu_score_pos = 0.
        cumu_score_neg = 0.
        ratio_score_pos_neg = 0.
        
        doc = nlp(text)
        doc_list = []
        sentences = list(doc.sents)
        word_list = []
        pos_tag_list = []
        token_list = []
        pos = []
        features = []

        number_sentence = len(sentences)
        
        for sentence in sentences:
            doc_sent = nlp(sentence.text)
            avg_len_sentence += len(list(doc_sent))
            for token in doc_sent :
                pos.append(token.pos_)
                word = token.text
                doc_list.append(word)
                lexeme = nlp.vocab[word]
                if lexeme.is_stop == False:
                    word_list.append(token.text)
                    pos_tag_list.append(token.pos_)
                    token_list.append(token)

        avg_len_sentence = avg_len_sentence/number_sentence

        for i in range(len(word_list)) :
            if pos_tag_list[i] in tagged_list :
                number_useful_word += 1 
                list_score = list(swn.senti_synsets(word_list[i], tagged_map[pos_tag_list[i]]))
                if len(list_score) > 0 : 
                    cumu_score_pos += list_score[0].pos_score()
                    cumu_score_neg += list_score[0].neg_score()
                    
        number_nom = pos.count('NOUN')
        number_nom_proper = pos.count('PROPN')
        number_adj = pos.count('ADJ')
        number_verb = pos.count('VERB')
        number_adv = pos.count('ADV')
        number_interjection = pos.count('INTJ')

        if cumu_score_neg == 0:
            cumu_score_neg = 0.1
            
        ratio_score_pos_neg = cumu_score_pos/cumu_score_neg

        features = [
            number_nom,
            number_nom_proper,
            number_adj,
            number_verb,
            number_adv,
            number_interjection,
            number_sentence,
            avg_len_sentence,
            number_useful_word,
            cumu_score_pos,
            cumu_score_neg,
            ratio_score_pos_neg
        ]
        
        X.append(features)
        y.append(is_pos)

    return X, y

def get_data_lis(dataset) :

    train_pos = load_reviews(reviews_dataset['train_pos_fn'])
    train_neg = load_reviews(reviews_dataset['train_neg_fn'])
    test_pos = load_reviews(reviews_dataset['test_pos_fn'])
    test_neg = load_reviews(reviews_dataset['test_neg_fn'])

    return train_pos, train_neg, test_pos, test_neg

def get_Jurafsky(dataset):

    train_pos, train_neg, test_pos, test_neg = get_data_lis(dataset)

    X_train_Jurafsky = []
    y_train_Jurafsky = []
    X_test_Jurafsky = []
    y_test_Jurafsky = []
    
    X_train_Jurafsky, y_train_Jurafsky = get_Jurafsky_X_y(X_train_Jurafsky, y_train_Jurafsky, train_pos, 1)
    X_train_Jurafsky, y_train_Jurafsky = get_Jurafsky_X_y(X_train_Jurafsky, y_train_Jurafsky, train_neg, 0)

    X_test_Jurafsky, y_test_Jurafsky = get_Jurafsky_X_y(X_test_Jurafsky, y_test_Jurafsky, test_pos, 1)
    X_test_Jurafsky, y_test_Jurafsky = get_Jurafsky_X_y(X_test_Jurafsky, y_test_Jurafsky, test_pos, 0)

    return X_train_Jurafsky, y_train_Jurafsky, X_test_Jurafsky, y_test_Jurafsky

def get_Ohana(dataset):

    train_pos, train_neg, test_pos, test_neg = get_data_lis(dataset)

    X_train_Ohana = []
    y_train_Ohana = []
    X_test_Ohana = []
    y_test_Ohana = []
    
    X_train_Ohana, y_train_Ohana = get_Ohana_X_y(X_train_Ohana, y_train_Ohana, train_pos, 1)
    X_train_Ohana, y_train_Ohana = get_Ohana_X_y(X_train_Ohana, y_train_Ohana, train_neg, 0)

    X_test_Ohana, y_test_Ohana = get_Ohana_X_y(X_test_Ohana, y_test_Ohana, test_pos, 1)
    X_test_Ohana, y_test_Ohana = get_Ohana_X_y(X_test_Ohana, y_test_Ohana, test_pos, 0)

    return X_train_Ohana, y_train_Ohana, X_test_Ohana, y_test_Ohana
    
def get_combined(dataset):

    X_train_combined = []
    y_train_combined = []
    X_test_combined = []
    y_test_combined = []
    
    X_train_Jurafsky, y_train_Jurafsky, X_test_Jurafsky, y_test_Jurafsky = get_Jurafsky(dataset)
    X_train_Ohana, y_train_Ohana, X_test_Ohana, y_test_Ohana = get_Ohana(dataset)

    for i in range(len(y_train_Jurafsky)) :
        X_train_combined.append(X_train_Jurafsky[i]+X_train_Ohana[i])

    for i in range(len(y_test_Jurafsky)) :
        X_test_combined.append(X_test_Jurafsky[i]+X_test_Ohana[i])

    y_train_combined = y_train_Jurafsky
    y_test_combined = y_test_Jurafsky

    return X_train_combined, y_train_combined, X_test_combined, y_test_combined    

def normalize(text):

    text = text.lower()
    text = nlp(text)
    lemmatized = list()
    
    for word in text:
        lemma = word.lemma_.strip()
        if lemma:
            lexeme = nlp.vocab[word.text]
            if lexeme.is_stop == False and word.pos_ != "PUNCT":
                lemmatized.append(lemma)           
    
    return " ".join(lemmatized)

def negated(text) :

    doc = nlp(text)
    sentences = list(doc.sents)
    negated = list()

    for sentence in sentences:
        negated_sentence = nc.convert_negated_words(sentence.text)
        negated.append(negated_sentence)

    return " ".join(negated)
    

def get_negated_words(dataset):

    train_pos, train_neg, test_pos, test_neg = get_data_lis(dataset)

    train = train_pos + train_neg
    test = test_pos + test_neg

    train = [negated(text) for text in train]
    test = [negated(text) for text in test]

    train = [normalize(text) for text in train]
    test = [normalize(text) for text in test]
    
    X_train_words = []
    y_train_words = []
    X_test_word = []
    y_test_words = []

    vectorizer = CountVectorizer(lowercase=True)

    X_train = pd.DataFrame(vectorizer.fit_transform(train).todense())
    X_test = pd.DataFrame(vectorizer.transform(test).todense())

    X_train_words = X_train
    X_test_words = X_test

    for i in range(len(train_pos)) :
        y_train_words.append(1)

    for i in range(len(train_neg)) :
        y_train_words.append(0)

    for i in range(len(test_pos)) :
        y_test_words.append(1)

    for i in range(len(test_neg)) :
        y_test_words.append(0)

    return X_train_words, y_train_words, X_test_words, y_test_words

def get_words(dataset):

    train_pos, train_neg, test_pos, test_neg = get_data_lis(dataset)

    train = train_pos + train_neg
    test = test_pos + test_neg

    train = [normalize(text) for text in train]
    test = [normalize(text) for text in test]
    
    X_train_words = []
    y_train_words = []
    X_test_word = []
    y_test_words = []

    vectorizer = CountVectorizer(lowercase=True)

    X_train = pd.DataFrame(vectorizer.fit_transform(train).todense())
    X_test = pd.DataFrame(vectorizer.transform(test).todense())

    X_train_words = X_train
    X_test_words = X_test

    for i in range(len(train_pos)) :
        y_train_words.append(1)

    for i in range(len(train_neg)) :
        y_train_words.append(0)

    for i in range(len(test_pos)) :
        y_test_words.append(1)

    for i in range(len(test_neg)) :
        y_test_words.append(0)

    return X_train_words, y_train_words, X_test_words, y_test_words
    
def get_results(y_train, y_train_pred, y_test, y_test_pred) :

    results = dict()
    results['accuracy_test'] = accuracy_score(y_test, y_test_pred)
    results['precision_test'] = precision_score(y_test, y_test_pred)
    results['recall_test'] = recall_score(y_test, y_test_pred)
    results['confusion_matrix_test'] = confusion_matrix(y_test, y_test_pred)

    results['accuracy_train'] = accuracy_score(y_train, y_train_pred)
    results['precision_train'] = precision_score(y_train, y_train_pred)
    results['recall_train'] = recall_score(y_train, y_train_pred)
    results['confusion_matrix_train'] = confusion_matrix(y_train, y_train_pred)

    return results

def get_train_test_data(dataset, features='jurafsky' ) :

    X_train = []
    y_train = []
    X_test = []
    y_test = []

    if features == 'jurafsky' :
        X_train, y_train, X_test, y_test = get_Jurafsky(dataset)
    if features == 'ohana' :
        X_train, y_train, X_test, y_test = get_Ohana(dataset)
    if features == 'combined' :
        X_train, y_train, X_test, y_test = get_combined(dataset)   
    if features == 'words' :
        X_train, y_train, X_test, y_test = get_words(dataset)
    if features == 'negated_words' :
        X_train, y_train, X_test, y_test = get_negated_words(dataset)

    return X_train, y_train, X_test, y_test
    

def NB(dataset, features='jurafsky') :
    classifier = MultinomialNB()

    X_train, y_train, X_test, y_test = get_train_test_data(dataset, features)

    classifier.fit(X_train, y_train)
    y_test_pred = classifier.predict(X_test)
    y_train_pred = classifier.predict(X_train)
    
    results = get_results(y_train, y_train_pred, y_test, y_test_pred)
        
    return results

def LG(dataset, features='jurafsky') :
    classifier = LogisticRegression(max_iter=400)

    X_train, y_train, X_test, y_test = get_train_test_data(dataset, features)

    classifier.fit(X_train, y_train)
    y_test_pred = classifier.predict(X_test)
    y_train_pred = classifier.predict(X_train)
    
    results = get_results(y_train, y_train_pred, y_test, y_test_pred)

    #np_X =np.array([np.array(xi) for xi in X_train])
    #print(np.std(np_X, 0)*classifier.coef_)
    
    return results

def NN(dataset, features='jurafsky') :
    classifier = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)

    X_train, y_train, X_test, y_test = get_train_test_data(dataset, features)

    classifier.fit(X_train, y_train)
    y_test_pred = classifier.predict(X_test)
    y_train_pred = classifier.predict(X_train)
    
    results = get_results(y_train, y_train_pred, y_test, y_test_pred)
        
    return results


def test_algorithm(dataset) :

    X_train_Jurafsky, y_train, X_test_Jurafsky, y_test = get_Jurafsky(dataset)
    X_train_Ohana, y_train, X_test_Ohana, y_test = get_Ohana(dataset)

    X_train_combined = list()
    X_test_combined = list()

    for i in range(len(y_train)) :
        X_train_combined.append(X_train_Jurafsky[i]+X_train_Ohana[i])

    for i in range(len(y_test)) :
        X_test_combined.append(X_test_Jurafsky[i]+X_test_Ohana[i])

    classifier_NB = MultinomialNB()
    classifier_LG = LogisticRegression(max_iter=500)
    classifier_NN = MLPClassifier(alpha=1e-3, max_iter=100, random_state=1)
    
    print("============ classifier_NB ===============")
    print("============ Jurafsky ===============")
    classifier_NB.fit(X_train_Jurafsky, y_train)
    y_test_pred = classifier_NB.predict(X_test_Jurafsky)
    y_train_pred = classifier_NB.predict(X_train_Jurafsky)
    results = get_results(y_train, y_train_pred, y_test, y_test_pred)
    print(results)
    print(classifier_NB.coef_)

    print("============ Ohana  ===============")
    classifier_NB.fit(X_train_Ohana , y_train)
    y_test_pred = classifier_NB.predict(X_test_Ohana )
    y_train_pred = classifier_NB.predict(X_train_Ohana )
    results = get_results(y_train, y_train_pred, y_test, y_test_pred)
    print(results)
    print(classifier_NB.coef_)

    print("============ combined ===============")
    classifier_NB.fit(X_train_combined, y_train)
    y_test_pred = classifier_NB.predict(X_test_combined)
    y_train_pred = classifier_NB.predict(X_train_combined)
    results = get_results(y_train, y_train_pred, y_test, y_test_pred)
    print(results)
    print(classifier_NB.coef_)

    print("============ classifier_LG ===============")
    print("============ Jurafsky ===============")
    classifier_LG.fit(X_train_Jurafsky, y_train)
    y_test_pred = classifier_LG.predict(X_test_Jurafsky)
    y_train_pred = classifier_LG.predict(X_train_Jurafsky)
    results = get_results(y_train, y_train_pred, y_test, y_test_pred)
    print(results)
    print(classifier_LG.coef_)

    print("============ Ohana  ===============")
    classifier_LG.fit(X_train_Ohana , y_train)
    y_test_pred = classifier_LG.predict(X_test_Ohana )
    y_train_pred = classifier_LG.predict(X_train_Ohana )
    results = get_results(y_train, y_train_pred, y_test, y_test_pred)
    print(results)
    print(classifier_LG.coef_)

    print("============ combined ===============")
    classifier_LG.fit(X_train_combined, y_train)
    y_test_pred = classifier_LG.predict(X_test_combined)
    y_train_pred = classifier_LG.predict(X_train_combined)
    results = get_results(y_train, y_train_pred, y_test, y_test_pred)
    print(results)
    print(classifier_LG.coef_)

    print("============ classifier_NN ===============")
    print("============ Jurafsky ===============")
    classifier_NN.fit(X_train_Jurafsky, y_train)
    y_test_pred = classifier_NN.predict(X_test_Jurafsky)
    y_train_pred = classifier_NN.predict(X_train_Jurafsky)
    results = get_results(y_train, y_train_pred, y_test, y_test_pred)
    print(results)

    print("============ Ohana  ===============")
    classifier_NN.fit(X_train_Ohana , y_train)
    y_test_pred = classifier_NN.predict(X_test_Ohana )
    y_train_pred = classifier_NN.predict(X_train_Ohana )
    results = get_results(y_train, y_train_pred, y_test, y_test_pred)
    print(results)

    print("============ combined ===============")
    classifier_NN.fit(X_train_combined, y_train)
    y_test_pred = classifier_NN.predict(X_test_combined)
    y_train_pred = classifier_NN.predict(X_train_combined)
    results = get_results(y_train, y_train_pred, y_test, y_test_pred)
    print(results)
   

    X_train_words, y_train, X_test_words, y_test = get_words(dataset)

    print("============ classifier_NB ===============")
    print("============ words ===============")
    classifier_NB.fit(X_train_words, y_train)
    y_test_pred = classifier_NB.predict(X_test_words)
    y_train_pred = classifier_NB.predict(X_train_words)
    results = get_results(y_train, y_train_pred, y_test, y_test_pred)
    print(results)


    print("============ classifier_LG ===============")
    print("============ words ===============")
    classifier_LG.fit(X_train_words, y_train)
    y_test_pred = classifier_LG.predict(X_test_words)
    y_train_pred = classifier_LG.predict(X_train_words)
    results = get_results(y_train, y_train_pred, y_test, y_test_pred)
    print(results)

    print("============ classifier_NN ===============")
    print("============ words ===============")
    classifier_NN.fit(X_train_words, y_train)
    y_test_pred = classifier_NN.predict(X_test_words)
    y_train_pred = classifier_NN.predict(X_train_words)
    results = get_results(y_train, y_train_pred, y_test, y_test_pred)
    print(results)
    
    X_train_negated_words, y_train, X_test_negated_words, y_test = get_negated_words(dataset)

    print("============ classifier_NB ===============")

    print("============ negated_words ===============")
    classifier_NB.fit(X_train_negated_words, y_train)
    y_test_pred = classifier_NB.predict(X_test_negated_words)
    y_train_pred = classifier_NB.predict(X_train_negated_words)
    results = get_results(y_train, y_train_pred, y_test, y_test_pred)
    print(results)

    print("============ classifier_LG ===============")

    print("============ negated_words ===============")
    classifier_LG.fit(X_train_negated_words, y_train)
    y_test_pred = classifier_LG.predict(X_test_negated_words)
    y_train_pred = classifier_LG.predict(X_train_negated_words)
    results = get_results(y_train, y_train_pred, y_test, y_test_pred)
    print(results)

    print("============ classifier_NN ===============")

    print("============ negated_words ===============")
    classifier_NN.fit(X_train_negated_words, y_train)
    y_test_pred = classifier_NN.predict(X_test_negated_words)
    y_train_pred = classifier_NN.predict(X_train_negated_words)
    results = get_results(y_train, y_train_pred, y_test, y_test_pred)
    print(results)
    

    

def train_and_test_classifier(dataset, model='NB', features='words'):
    """
    :param dataset: les 4 fichiers utilisées pour entraîner et tester les classificateurs. Voir reviews_dataset.
    :param model: le type de classificateur. NB = Naive Bayes, LG = Régression logistique, NN = réseau de neurones
    :param features: le type d'attributs (features) que votre programme doit construire
                 - 'jurafsky': les 6 attributs proposés dans le livre de Jurafsky et Martin.
                 - 'ohana': les 12 attributs représentant le style de rédaction (Ohana et al.)
                 - 'combined': tous les attributs 'jurafsky' et 'ohaha'
                 - 'words': des vecteurs de mots
                 - 'negated_words': des vecteur de mots avec conversion des mots dans la portée d'une négation
    :return: un dictionnaire contenant 3 valeurs:
                 - l'accuracy à l'entraînement (validation croisée)
                 - l'accuracy sur le jeu de test
                 - la matrice de confusion obtenu de scikit-learn

    """
    

    #test_algorithm(dataset)
    
    all_results = dict()
    results = dict()

    if model=='NB':
        if features=='jurafsky' :
            all_results = NB(dataset, 'jurafsky')
        if features=='ohana' :
            all_results = NB(dataset, 'ohana')
        if features=='combined' :
            all_results = NB(dataset, 'combined')
        if features=='words' :
            all_results = NB(dataset, 'words')
        if features=='negated_words' :
            all_results = NB(dataset, 'negated_words')

    if model=='LG':
        if features=='jurafsky' :
            all_results = LG(dataset, 'jurafsky')
        if features=='ohana' :
            all_results = LG(dataset, 'ohana')
        if features=='combined' :
            all_results = LG(dataset, 'combined')
        if features=='words' :
            all_results = LG(dataset, 'words')
        if features=='negated_words' :
            all_results = LG(dataset, 'negated_words')

    if model=='NN':
        if features=='jurafsky' :
            all_results = NN(dataset, 'jurafsky')
        if features=='ohana' :
            all_results = NN(dataset, 'ohana')
        if features=='combined' :
            all_results = NN(dataset, 'combined')
        if features=='words' :
            all_results = NN(dataset, 'words')
        if features=='negated_words' :
            all_results = NN(dataset, 'negated_words')

    
    # Les résultats à retourner 
    results['accuracy_train'] = all_results['accuracy_train']
    results['accuracy_test'] = all_results['accuracy_test']
    results['confusion_matrix'] = all_results['confusion_matrix_test']  # la matrice de confusion obtenue de Scikit-learn
    return results


def load_reviews(filename):
    with open(filename, 'r') as fp:
        reviews_list = json.load(fp)
    return reviews_list


if __name__ == '__main__':

    results = train_and_test_classifier(reviews_dataset, model='LG', features='ohana')   
    print("Accuracy - entraînement: ", results['accuracy_train'])
    print("Accuracy - test: ", results['accuracy_test'])
    print("Matrice de confusion: ", results['confusion_matrix'])


