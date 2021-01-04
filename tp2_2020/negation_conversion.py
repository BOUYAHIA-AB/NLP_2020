# -*- coding: utf-8 -*-
import json
import spacy

nlp = spacy.load("en_core_web_sm")

example_fn = "./data/exemples_t1.json"
#example_fn = "./data/test1_t1.json"

other_negation_words = ["No", "no", "hardly", "barely", "none", "nothing", "without"]

def load_examples(filename):
    with open(filename, 'r') as fp:
        example_list = json.load(fp)
    return example_list

def find_lca(sentence) : 
    doc = nlp(sentence)
    negation_tokens = [tok for tok in doc if tok.dep_ == 'neg']
    for token in doc :
        if token.text in other_negation_words :
            negation_tokens.append(token)


    negation_tokens_dict = dict()
    for tok in negation_tokens :
        negation_tokens_dict[tok.i] = tok

    matrix = doc.get_lca_matrix()

    lca = dict()
    
    for key, value in negation_tokens_dict.items():
        if key < len(doc) - 1 :
            index = matrix[key,key+1]
            lca[index] = key
    
    return negation_tokens_dict, lca

def ComputedCandidateScope(sentence) :
    negation_tokens_dict, lca = find_lca(sentence)
    doc = nlp(sentence)

    CandidateScope = dict()

    for key, value in lca.items() :
        word = doc[key]
        CandidateScope_dict = dict()
        for w in word.subtree :
            if w.i > value :
                CandidateScope_dict[w.i]= w.text

        CandidateScope[value] = CandidateScope_dict     
    
    return CandidateScope

def get_scope(sentence) :
    CandidateScope = ComputedCandidateScope(sentence)
    scope = dict()
    delimiters = ["when", "whenever", "whether","because", "unless", "since", "hence", "while"]
    condi_delimiters = ["so","as", "even", "which","who", "why", "where", "for", "like","but",".",",",";",":","!","?"]

    

    for key, value in CandidateScope.items() :
        first_element_in_value = value[next(iter(value))]

        if first_element_in_value in condi_delimiters :
            condi_delimiters.remove(first_element_in_value)

        scope_dict = dict()

        for key_1, value_1 in value.items() :
            if value_1 in condi_delimiters + delimiters:
                break
            scope_dict[key_1] = value_1

        scope[key] = scope_dict

    return scope
    
def get_negation(sentence):
    scope_dict = get_scope(sentence)
    new_doc = []
    doc = nlp(sentence)
    i=0
    exception =['.',',','"','_','-','(',')','[',']']

    scope = dict() 
    for key, value in scope_dict.items() :
        for key_1, value_1 in value.items() :
            scope[key_1] = value_1

    for token in doc :
        if token.i in scope.keys() and token.pos_ != "AUX" and token.text not in exception:
            new_doc.append('NOT\_'+token.text)
            i+=1
        else :
            new_doc.append(token.text)
    
    return ' '.join(new_doc)
    
def is_exception(sentence):
    doc = nlp(sentence)
    negation_tokens = [tok for tok in doc if tok.dep_ == 'neg']
    
    for token in doc :
        if (token.text in other_negation_words) and len(negation_tokens)==0 and token.i < len(doc) - 1:
            negation_tokens.append(token)
            return False

    if len(negation_tokens)==0 :
        return True

    i = 0
    for token in doc:
        if token in negation_tokens :
            break
        i+=1

    if i > len(doc)-2 :
        return True
    
    if doc[i+1].text == "just" or doc[i+1].text == "wonder" :
        return True

    return False
    

def convert_negated_words(sentence):
    # Voir l'énoncé du travail concernant la tâche de détection de la portée de la négation.
    # SVP ne pas changer la signature de la fonction.
    #
    # Pour déterminer la portée d'une négation :
    #  - utilisez spacy
    #  - vous devez utiliser la structure du graphe de dépendances,
    #  - si cela est utile, vous pouvez utiliser les part-of-speech (POS) ou d'autres informations.
    #
    # Mettre votre code ici. Vous pouvez effacer les commentaires.

    converted_sentence = sentence
    if not is_exception(sentence) :
        converted_sentence = get_negation(sentence)  # A MODIFIER

    return converted_sentence.strip()

def test_examples(filename):
    examples = load_examples(filename)
    i=1
    for example in examples:
        print("\nPhrase: ",i," : ", example['S'])
        i+=1
        converted = convert_negated_words(example['S'])
        print("Conversion:", converted)
        print("Solution:  ", example['N'])

if __name__ == '__main__':
    
    test_examples(example_fn)
