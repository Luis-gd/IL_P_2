import pandas as pd
import spacy
import es_core_news_sm
from spacy.matcher import Matcher
import re

# Elección del CV a analizar
rutas = ["cv_txt/cv_jorge.txt", "cv_txt/cv_luis.txt"]

indice = int(input("Escribe 0 para elegir el curriculum de jorge o 1 para elegir el curriculum de luis: "))

while (indice not in range(2)):
    indice = int(input("Escribe 0 para elegir el curriculum de jorge o 1 para elegir el curriculum de luis: "))

fichero = rutas[indice]

#Cambiamos la variable pregunta para elegir qué dato tomamos

pregunta = 1

# ,pregunta = input("Escriba su pregunta: ")

# Procesamiento del txt

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

# Extracción del nombre del candidato:

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

def devolver_nombre(x):
    nombre = None
    for f in range(len(df['text'])):
        if df['Nombre'][f] != None and df['Apellido1'][f] != None and df['Apellido2'][f] != None:
            if f == 0:
                nombre = df['Nombre'][f] + ' ' + df['Apellido1'][f] + ' ' + df['Apellido2'][f] 
    return nombre

last_token = ['\n']

pattern_father = [[{'POS':'PROPN', 'OP' : '+'},
           {'POS':'PROPN', 'OP' : '+'},
           {'POS':'PROPN', 'OP' : '+'},
           {'LOWER': {'IN' : last_token}}]]

if pregunta == 0:
    new_columns = ['Nombre','Apellido1', 'Apellido2']
    for n,col in enumerate(new_columns):
        df[col] = df['text'].apply(lambda x: encontrar_nombre(x)).apply(lambda x: x[n])



# Prueba de la función devolver nombre y apellidos:

# print(devolver_nombre(df))

# Extracción de correos:
'''
data = open(fichero,'r')
texto = data.read() 

r = re.compile(r'(\b[\w.]+@+[\w.]+.+[\w.]\b)')
results = r.findall(texto)
email_del_candidato = results[0]

 # Esto nos da todos los emails que contiene el texto, 

 '''
'''
new_columns = ['Número']
for n,col in enumerate(new_columns):
    df[col] = df['text'].apply(lambda x: encontrar_numero(x)).apply(lambda x: x[n]) '''

# Extracción de secciones:

secciones = ['EXPERIENCIA', 'EDUCACIÓN', 'CONTACTO', 'HABILIDADES', 'IDIOMAS', 'REFERENCIAS']

def encontrar_seccion(seccion):
    fila_i = None
    fila_f = None
    for f in range(len(df['text'])):
        if seccion in df['text'][f]:
            fila_i = f
        v = [l.isupper() for l in df['text'][f].strip('\n')]
        if fila_i != None and all(v) and f!= fila_i and len(v) != 0:
            fila_f = f
            break
    return (fila_i, fila_f) 
    
'''
if pregunta == 1:
    new_columns = ['Experiencia']
    for n,col in enumerate(new_columns):
        df[col] = df['text'].apply(lambda x: encontrar_seccion_experiencia(x)).apply(lambda x: x[n])
'''
print(encontrar_seccion(secciones[0]))
def saca_texto(lineas):
    i, f = lineas
    texto = ''
    for j in range(i, f):
        texto += df['text'][j]
    return texto

print(saca_texto(encontrar_seccion(secciones[0])))