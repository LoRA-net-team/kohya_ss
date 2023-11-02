import os

text_file_dir = r'sentence_datas/teddy_bear_sentence_200.txt'
with open(text_file_dir, 'r', encoding='utf-8') as f:
    lines = f.readlines()
new_lines = []
for line in lines :
    line = line.strip()
    if line not in new_lines :
        new_lines.append(line)

new_text_file_dir = f'sentence_datas/teddy_bear_sentence_{len(new_lines)}.txt'
with open(new_text_file_dir, 'w', encoding='utf-8') as f :
    for line in new_lines :
        f.write(line+'\n')