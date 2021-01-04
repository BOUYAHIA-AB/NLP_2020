import json
from nltk import word_tokenize, bigrams, trigrams
from nltk.util import ngrams
from nltk.lm import MLE
from nltk.lm.models import Laplace
from nltk.util import pad_sequence
from operator import itemgetter
import re
import pandas as pd
import numpy
from collections import Counter, defaultdict
import math

proverbs_fn = "./data/proverbes.txt"
test1_fn = "./data/test_proverbes1.txt"

BOS = '<BOS>'
EOS = '<EOS>'
model_1grams = Laplace(1)
model_2grams = Laplace(2)
model_3grams = Laplace(3)
#model = defaultdict(lambda: defaultdict(lambda: 0))

def load_proverbs(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        raw_lines = f.readlines()
    return [x.strip() for x in raw_lines]


def load_tests(filename):
    with open(filename, 'r') as fp:
        test_data = json.load(fp)
    return test_data

def build_vocabulary(text_list):
    all_unigrams = list()
    for sentence in text_list:
        word_list = word_tokenize(sentence)
        all_unigrams = all_unigrams + word_list
    voc = set(all_unigrams)
    voc.add(BOS)
    voc.add(EOS)
    return list(voc)

def get_ngrams(text_list, n=2):
    all_ngrams = list()
    for sentence in text_list:
        tokens = word_tokenize(sentence)
        padded_sent = list(pad_sequence(tokens, pad_left=True, left_pad_symbol=BOS, pad_right=True, right_pad_symbol=EOS, n=n))
        all_ngrams = all_ngrams + list(ngrams(padded_sent, n=n))      
    return all_ngrams

class my_model_bigramme(object):

    def __init__(self) :
        self.model = defaultdict(lambda: defaultdict(lambda: 0))

    def train(self, corpus_2grams, vocabulary) :
        self.corpus_2grams = corpus_2grams
        self.vocabulary = vocabulary

        for word1, word2 in self.corpus_2grams:
            self.model[word1][word2] += 1

        #lissage de laplace

        for word1 in self.vocabulary :
            for word2 in self.vocabulary :
                self.model[word1][word2] += 1    

        for word1 in self.model:
            total_count = float(sum(self.model[word1].values()))
            for word2 in self.model[word1]:
                self.model[word1][word2] /= total_count

    def logprop_bigramme(self, proverb_to_estime) :
        list_UNK = list()
        tokens = word_tokenize(proverb_to_estime)
        for word in tokens:
            if word not in self.vocabulary :
                proverb_to_estime = re.sub(word, "", proverb_to_estime)

        proverb_2grams = get_ngrams([proverb_to_estime], n=2)

        logprop = 0
        for word1, word2 in proverb_2grams :
            logprop+= math.log(self.model[word1][word2],2)

        return logprop

model_bigramme = my_model_bigramme()
    
def train_models(filename):
    proverbs = load_proverbs(filename)
    """ Vous ajoutez à partir d'ici tout le code dont vous avez besoin
        pour construire les différents modèles N-grammes.
        Voir les consignes de l'énoncé du travail pratique concernant les modèles à entraîner.

        Vous pouvez ajouter au fichier toutes les fonctions, classes, méthodes et variables que vous jugerez nécessaire.
        Merci de ne pas modifier les signatures (noms de fonctions et arguments) déjà présentes dans le fichier.
    """

    # Votre code à partir d'ici...
    
    vocabulary = build_vocabulary(proverbs)
    corpus_1grams = get_ngrams(proverbs, n=1)
    corpus_2grams = get_ngrams(proverbs, n=2)
    corpus_3grams = get_ngrams(proverbs, n=3)

    model_1grams.fit([corpus_1grams], vocabulary_text=vocabulary)
    model_2grams.fit([corpus_2grams], vocabulary_text=vocabulary)
    model_3grams.fit([corpus_3grams], vocabulary_text=vocabulary)
    # my_model_bigramme(corpus_2grams = corpus_2grams , vocabulary = vocabulary)
    model_bigramme.train(corpus_2grams, vocabulary)

def cloze_test(incomplete_proverb, choices, n=3):
    """ Fonction qui complète un texte à trous en ajoutant le(s) bon(s) mot(s).
        En anglais, on nomme ce type de tâche un cloze test.

        La paramètre n désigne le modèle utilisé.
        1 - unigramme NLTK, 2 - bigramme NLTK, 3 - trigramme NLTK, 20 - votre modèle bigramme
    """
    
    # Votre code à partir d'ici.
    
    perplexities = list()
    logprops = list()
    index = 0
    element = choices[0]
    condidat_proverb = ''

    for choice in choices :
        all_ngrams = list()
        condidat_proverb = re.sub('\*\*\*', choice, incomplete_proverb)
        
        if n == 1:
            all_ngrams = all_ngrams + get_ngrams([condidat_proverb], n=n)
            perplexities.append(model_1grams.perplexity(all_ngrams))      
        elif n == 2 :
            all_ngrams = all_ngrams + get_ngrams([condidat_proverb], n=2)
            perplexities.append(model_2grams.perplexity(all_ngrams))
        elif n == 3 :
            all_ngrams = all_ngrams + get_ngrams([condidat_proverb], n=3)
            perplexities.append(model_3grams.perplexity(all_ngrams))
        elif n == 20 :
            logprops.append(model_bigramme.logprop_bigramme(condidat_proverb))            
    
    if n==1 or n==2 or n==3 :
        index, element = min(enumerate(perplexities), key=itemgetter(1))
    elif n==20 :
        index, element = max(enumerate(logprops), key=itemgetter(1))
        
    mot_remplace = choices[index]
   
    result = re.sub('\*\*\*', mot_remplace, incomplete_proverb)
    perplexity = element
    return result, perplexity

def logpop_NLTK(proverb_to_estime):
    all_ngrams = list()
    all_ngrams = all_ngrams + get_ngrams([proverb_to_estime], n=2)
    logprop = 0
    for word1, word2 in all_ngrams :
        logprop+= model_2grams.logscore(word2, [word1])
        
    return logprop
    
    

if __name__ == '__main__':
    # Vous pouvez modifier cette section comme bon vous semble
    proverbs = load_proverbs(proverbs_fn)
    print("\nNombre de proverbes : ", len(proverbs))
    train_models(proverbs_fn)

    """
    print(model_bigramme.logprop_bigramme("a beau mentir qui vient de loin"))
    print(model_bigramme.logprop_bigramme("a beau se lever tard, qui a bruit de se lever matin"))
    print(model_bigramme.logprop_bigramme("abandon fait larron"))
    print(model_bigramme.logprop_bigramme("abondance de biens ne nuit pas"))
    print(model_bigramme.logprop_bigramme("accord vaut mieux qu’argent"))
    print(model_bigramme.logprop_bigramme("amour fait beaucoup, mais argent fait tout"))

    print("============================================")
    print(logpop_NLTK("a beau mentir qui vient de loin"))
    print(logpop_NLTK("a beau se lever tard, qui a bruit de se lever matin"))
    print(logpop_NLTK("abandon fait larron"))
    print(logpop_NLTK("abondance de biens ne nuit pas"))
    print(logpop_NLTK("accord vaut mieux qu’argent"))
    print(logpop_NLTK("amour fait beaucoup, mais argent fait tout"))

    """

    #log_propa = logprop_bigramme("a beau mentir qui programme de loin")

    #print("log_propa : ", log_propa)

    f = open("./data/proverbe_predit.txt", "w")
           
    test_proverbs = load_tests(test1_fn)
    print("\nNombre de tests du fichier {}: {}\n".format(test1_fn, len(test_proverbs)))
    print("Les résultats des tests sont:")
    for partial_proverb, options in test_proverbs.items():
        solution, perplexity = cloze_test(partial_proverb, options, n=20)
        print("\n\tProverbe incomplet: {} , Options: {}".format(partial_proverb, options))
        print("\tSolution = {} , Perplexité = {}".format(solution, perplexity))
        f.write("{}\n".format(solution))
    

    
    
    

    
