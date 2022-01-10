import pandas as pd
import spacy
import es_core_news_sm
from spacy.matcher import Matcher

rutas = ["cv_txt/cv_jorge.txt", "cv_txt/cv_luis.txt"]

indice = int(input("Escribe 0 para elegir el curriculum de jorge o 1 para elegir el curriculum de luis: "))

while (indice not in range(2)):
    indice = int(input("Escribe 0 para elegir el curriculum de jorge o 1 para elegir el curriculum de luis: "))

fichero = rutas[indice]

# ,pregunta = input("Escriba su pregunta: ")

with open(fichero, 'r') as f:
    text = [line for line in f.readlines()]

df = pd.DataFrame(text,columns=['text'])
df.head()

text = df['text'][0]
nlp = es_core_news_sm.load()
doc = nlp(text)

features = []
for token in doc:
    features.append({'token' : token.text, 'pos' : token.pos_})

fdf = pd.DataFrame(features)
fdf.head(len(fdf))

first_tokens = ['to', 'father']
last_tokens = ['and', 'naming']

pattern_father = [[{'POS':'PROPN', 'OP' : '+'},
           {'POS':'PROPN', 'OP' : '+'},
           {'POS':'PROPN', 'OP' : '+'}]]

def encontrar_nombre(x):
    nlp = es_core_news_sm.load()
    doc = nlp(x)
    matcher = Matcher(nlp.vocab) 
    matcher.add("matching_father", pattern_father)
    matches = matcher(doc)
    sub_text = ''    
    if(len(matches) > 0):
        span = doc[matches[0][1]:matches[0][2]] 
        sub_text = span.text
    tokens = sub_text.split(' ')

    if len(tokens) == 3:
        name, surname1, surname2 = tokens[0], tokens[1], tokens[2]
    else:
        name, surname1, surname2 = None, None, None
    return name, surname1, surname2

new_columns = ['Nombre','Apellido1', 'Apellido2']
for n,col in enumerate(new_columns):
    df[col] = df['text'].apply(lambda x: encontrar_nombre(x)).apply(lambda x: x[n])

print(df)
