import pandas as pd
import spacy
import es_core_news_sm
from spacy.matcher import Matcher
import re
import unicodedata

secciones = ['EXPERIENCIA', 'EDUCACIÓN', 'CONTACTO', 'HABILIDADES', 'IDIOMAS', 'REFERENCIAS']

# Extracción de secciones:


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

def saca_texto(lineas):
    i, f = lineas
    texto = ''
    for j in range(i, f):
        texto += df['text'][j]
    return texto

# print(saca_texto(encontrar_seccion(secciones[0])))

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


# Extracción de lenguajes de programación:

def encontrar_lenguajes():
    lenguajes = 'Python, R, Java, C, JavaScript, Matlab, SQL, SPARQL, C++'
    punto = None
    fila = None
    for f in range(len(df['text'])):
        if 'lenguajes de programación' in df['text'][f].lower():
            fila = f
            punto = df['text'][f].lower().index('lenguajes')
            break
    cadena = df['text'][fila][punto:].replace(',','').replace('.','').split()
    l = [i for i in cadena if i in lenguajes.replace(',','').split()]
    return ', '.join(l)


def main():
    parada = False
    while (not parada):
        # Elección del CV a analizar
        rutas = ["cv_txt/cv_jorge.txt", "cv_txt/cv_luis.txt"]

        indice = input(("Escriba 0 para elegir el primer currículum, "
        "1 para elegir el segundo currículum o "
        "cualquier otra cosa para salir del sistema de preguntas:"))

        if indice.isdigit():
            if indice not in range(2):
                print("Gracias por hacer uso de este servicio. Le esperamos cuando tenga alguna duda.")
                parada = True
                break
            else:
                indice = int(indice) 
        else:
            print("Gracias por hacer uso de este servicio. Le esperamos cuando tenga alguna duda.")
            parada = True
            break

        fichero = rutas[indice]

        #Cambiamos la variable pregunta para elegir qué dato tomamos
        # Las preguntas está hechas de esta forma hasta que sepamos qué preguntas base y qué parafraseo poner. 
        # Los números llegan hasta el 7, que son las dimensiones que tenemos actualmente. 
        # Si queremos añadir más solo habría que aumentar el range

        pregunta = input(("¿Qué quieres saber sobre el candidato? \n"
            "Para saber su nombre completo introduce 0. \n"
            "Para saber su correo electrónico pulsa 1. \n"
            "Para saber los idiomas que maneja pulsa 3. \n"
        ))

        if pregunta.isdigit():
            if pregunta not in range(2):
                print("Gracias por hacer uso de este servicio. Le esperamos cuando tenga alguna duda.")
                parada = True
                break
            else:
                pregunta = int(pregunta) 
        else:
            print("Gracias por hacer uso de este servicio. Le esperamos cuando tenga alguna duda.")
            parada = True
            break

        while pregunta not in range(8):
            
            pregunta = int(input(("¿Qué quieres saber sobre el candidato? \n"
                "Para saber su nombre completo introduce 0. \n"
                "Para saber su correo electrónico pulsa 1. \n"
                "Para saber los idiomas que maneja pulsa 3. \n"
            )))
            if pregunta.isdigit():
                if pregunta not in range(2):
                    print("Gracias por hacer uso de este servicio. Le esperamos cuando tenga alguna duda.")
                    parada = True
                    break
                else:
                    pregunta = int(pregunta) 
            else:
                print("Gracias por hacer uso de este servicio. Le esperamos cuando tenga alguna duda.")
                parada = True
                break

            
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

        # Llamada para sacar el nombre del candidato
        if pregunta == 0:
            new_columns = ['Nombre','Apellido1', 'Apellido2']
            for n,col in enumerate(new_columns):
                df[col] = df['text'].apply(lambda x: encontrar_nombre(x)).apply(lambda x: x[n])
            print("El nombre completo del candidato es:", devolver_nombre(df))

        # Llamada para sacar el correo del candidato
        if pregunta == 1:
            data = open(fichero,'r')
            texto = data.read() 

            r = re.compile(r'(\b[\w.]+@+[\w.]+.+[\w.]\b)')
            results = r.findall(texto)
            email_del_candidato = results[0]
            print("El email del candidato es:", email_del_candidato)

        # Llamada para sacar el número de teléfono del candidato
        # Te dejo la estructura del email porque podría sacarse de forma similar
        if pregunta == 2:
            data = open(fichero,'r')
            texto = data.read() 

            r = re.compile(r'(\b[\w.]+@+[\w.]+.+[\w.]\b)')
            results = r.findall(texto)
            email_del_candidato = results[0]
            print("El teléfono del candidato es: ESTÁ POR HACER")

        # Llamada a sacar los lenguajes de programación que conoce el candidato
        if pregunta == 3:
            print('Los lenguajes de programación que maneja el candidato son:', encontrar_lenguajes() + '.') 

        # Llamada para sacar los idiomas que sabe el candidato
        if pregunta == 4:
            print("Los idiomas que maneja el candidato son:", encontrar_seccion(secciones[4]))

        # Llamada para sacar las referencias del candidato
        if pregunta == 5:
            print("Las referencias del candidato son:", encontrar_seccion(secciones[5]))

        # Llamada para sacar la información académica del candidato
        if pregunta == 6:
            print("La formación académica del candidato es:", encontrar_seccion(secciones[1]))

        # Llamada para sacar la información académica del candidato
        if pregunta == 7:
            print("La experiencia laboral del candidato es:", encontrar_seccion(secciones[0]))

main()