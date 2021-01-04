# -*- coding: utf-8 -*-
import re

ingredients_fn = "./data/ingredients.txt"

# Mettre dans cette partie la (les) expression(s) régulière(s)
# que vous utilisez pour analyser les ingrédients
#
# Vos regex ici...
#

pattern_tasse_cuilliere_wasf = r'\b(?P<quantite>((\d+[,/\.]?)+|Une|[\u00BC-\u00BE\u2150-\u215E\u2189]+)(\s+à\s+(\d+[,/]?))?\s+(noix|cuillère|tasse|quinzaine|feuille|c\.|cl|tranche|botte|verre|tronc|tronçon|gousse|gallon|lb|pincée|pinte|enveloppe|oz|morceau|[bB]ouquet|[Rr]ondelle|lamelle)s?(\s+à\s+(café|soupe|thé|[cs]\.|\.[cs]))?(\s+\(\d+\s+ml\))?)(\s+d[’e\']\s?|\s)(?P<ingredient>[’\w\'\%\s]+)(\s+(d’environ|rincés|égouttées|désossées et coupées|rattes coupée?s|coupés|en purée|râpée?|hachées)([’\w\'\%\s]+)?)(,[,’\w\'\%\s]+)?'
pattern_tasse_cuilliere = r'\b(?P<quantite>((\d+[,/\.]?)+|Une|une petite|[\u00BC-\u00BE\u2150-\u215E\u2189]+)(\s+à\s+(\d+[,/]?))?\s+(noix|cuillère|tasse|quinzaine|feuille|c\.|cl|tranche|botte|verre|tronc|tronçon|gousse|gallon|lb|pincée|pinte|enveloppe|oz|morceau|[bB]ouquet|[Rr]ondelle|lamelle)s?(\s+à\s+(café|soupe|thé|[cs]\.|\.[cs]))?(\s+\(\d+\s+ml\))?)(\s+d[’e\']\s?|\s+)(?P<ingredient>[’\w\'\%\s]+)(,[,’\w\'\%\s]+)?'
pattern_glacon = r'\b(?P<quantite>((\d+[,/]?)+|[\u00BC-\u00BE\u2150-\u215E\u2189]+))\s+(?P<ingredient>glaçons?)([’\w\s\'\%]+)?'
pattern_ml_g_wasf = r'\b(?P<quantite>((\d+[,/]?)+|[\u00BC-\u00BE\u2150-\u215E\u2189]+)\s+(mL|ml|g|kg|boîtes?|c\.à\.[sc])s?(\s+\(((\d+[,/]?)+|[\u00BC-\u00BE\u2150-\u215E\u2189]+)(\s[\u00BC-\u00BE\u2150-\u215E\u2189]+)?[\w\'\.\%\s]+\))?)(\sd[e’\']\s?|\s)?(?P<ingredient>[’\w\'\%\s]+)(\s+(rattes\s)?(rincés|égouttées|torréfiées|concassés|ciselé|et décongelées|coupée?s|hachées?|tranché|en purée|frais moulu|râpé)([’\w\'\%\s]+)?)(,[,’\w\'\%\s]+)?'
pattern_ml_g = r'\b(?P<quantite>((\d+[,/]?)+|[\u00BC-\u00BE\u2150-\u215E\u2189]+)\s+(mL|ml|g|kg|boîtes?(\s+de conserve)?|c\.à\.[sc])s?(\s+\(((\d+[,/]?)+|[\u00BC-\u00BE\u2150-\u215E\u2189]+)(\s[\u00BC-\u00BE\u2150-\u215E\u2189]+)?[\w\'\.\%\s]+\))?)(\sd[e’\']\s?|\s)?(?P<ingredient>[’\w\'\%\s]+)(,[,’\w\'\%\s]+)?'
pattern_fruit_wasf = r'\b(?P<quantite>((\d+[,/]?)+|[\u00BC-\u00BE\u2150-\u215E\u2189]+))(\sd[e’\']\s?|\s)?(?P<ingredient>[’\w\'\%]+)\s+(pelées|en purée|(coupées?|hachés?) finement|émincés?|battu|tranché)(,[,’\w\'\%\s]+)?'
pattern_fruit = r'\b(?P<quantite>((\d+[,/]?)+|Zeste|trait|[\u00BC-\u00BE\u2150-\u215E\u2189]+)(\s+à\s+(\d+[,/]?))?)(\sd[e’\']\s?|\s)?(?P<ingredient>[’\w\s\'\%]+)'
pattern_huile_poivre = r'\b(?P<ingredient>([hH]uile|[Pp]oivre)[’\s\w\'\%]+)(?P<quantite>)'
pattern_sel_gout = r'\b(?P<ingredient>((du\s+)?([sS]el|eau|ail|clou et cannelle)(\s+(et|du) poivre(\s+du moulin)?)?))\s+(?P<quantite>au goût)'
pattern_sel = r'\b(?P<ingredient>((du\s+)?([sS]el|eau|ail|[Cc]arottes)(\s+(et|du) poivre(\s+du moulin)?)?))(?P<quantite>)'
pattern_feuille = r'\b(?P<quantite>([qQ]uelques?\s+)?([fF]euilles?|sommités?))(\sd[e’\']\s?|\s)?(?P<ingredient>[’\w\s\'\%]+)'

all_regex = [("pattern_1", pattern_tasse_cuilliere_wasf),
             ("pattern_2", pattern_tasse_cuilliere),
             ("pattern_3", pattern_glacon),
             ("pattern_4", pattern_ml_g_wasf),
             ("pattern_5", pattern_ml_g),
             ("pattern_6", pattern_fruit_wasf),
             ("pattern_7", pattern_fruit),
             ("pattern_8", pattern_huile_poivre),
             ("pattern_9", pattern_sel_gout),
             ("pattern_10", pattern_sel),
             ("pattern_11", pattern_feuille)            
            ]

def load_ingredients(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        raw_items = f.readlines()
    ingredients = [x.strip() for x in raw_items]
    return ingredients


def get_ingredients(text):
    # Insérez ici votre code pour l'extaction d'ingrédients.
    # En entrée, on devrait recevoir une ligne de texte qui correspond à un ingrédient.
    # Par ex. 2 cuillères à café de poudre à pâte
    # Vous pouvez ajouter autant de fonctions que vous le souhaitez.
    #
    # IMPORTANT : Ne pas modifier la signature de cette fonction
    #             afin de faciliter notre travail de correction.
    #
    # Votre code ici...
    #
    ingredient = ""
    quantite = ""
    i=0

    for tag, regex in all_regex:
        pattern = re.compile(regex)
        match = pattern.match(text)
        if match :
            ingredient = match.group('ingredient')
            quantite = match.group('quantite')
            return quantite, ingredient
        

    return quantite, ingredient   # À modifier - retourner la paire extraite



if __name__ == '__main__':
    # Vous pouvez modifier cette section
    print("Lecture des ingrédients du fichier {}. Voici quelques exemples: ".format(ingredients_fn))
    all_items = load_ingredients(ingredients_fn)
    f = open("./data/demofile.txt", "w")
    
    print("\nExemples d'extraction")
    for item in all_items:
        quantity, ingredient = get_ingredients(item)
        print("\t{}\t QUANTITE: {}\t INGREDIENT: {}".format(item, quantity, ingredient))
        f.write("{}   QUANTITE:{}   INGREDIENT:{}\n".format(item, quantity, ingredient))

    f.close()
