clear
python t2_completer_proverbes.py
grep -vf data/proverbe_correct.txt data/proverbe_predit.txt  >data/non_conforme_proverbe.txt
wc -l data/non_conforme_proverbe.txt