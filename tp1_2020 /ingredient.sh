clear
python t1_extraction_ingredients.py
awk 'a[$0]++' data/demofile.txt data/ingredients_solutions.txt | wc -l
grep -vf data/demofile.txt data/ingredients_solutions.txt >data/non_conforme.txt
Wc -l data/ingredients.txt
wc -l data/non_conforme.txt

