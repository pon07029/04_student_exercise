import json

with open('c:/Users/SNUVBMI5/Desktop/JHJ/AIChatVet/04_student_exercise/00_AIChatVet_Full_Course.ipynb', 'r', encoding='utf-8') as f:
    n = json.load(f)

with open('c:/Users/SNUVBMI5/Desktop/JHJ/AIChatVet/04_student_exercise/nb_dump.txt', 'w', encoding='utf-8') as f:
    for i, c in enumerate(n['cells']):
        f.write(f"\n---CELL {i} ({c.get('cell_type', '')})---\n")
        f.write(''.join(c.get('source', [])) + "\n")
