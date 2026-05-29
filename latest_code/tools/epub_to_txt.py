import zipfile
import os
from bs4 import BeautifulSoup

epub_path = "/Users/terrysun/Downloads/victor-hugo_notre-dame-de-paris_isabel-f-hapgood.epub"
txt_path = "Notre-Dame de Paris.txt"

with zipfile.ZipFile(epub_path, 'r') as z:
    for file in z.namelist():
        if file.endswith('.xhtml') or file.endswith('.html'):
            with z.open(file) as f:
                soup = BeautifulSoup(f, 'html.parser')
                text = soup.get_text()
                with open(txt_path, 'a', encoding='utf-8') as out:
                    out.write(text + '\n\n')