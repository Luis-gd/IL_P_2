#Imports
import os

#Read files
train = []
test = []

os.path

for i in range(30):
    #Deporte
    path = os.path.dirname(__file__)+"/dataset/deportes/deportes"+str(2*i+1)+".txt"
    with open(path) as f:
         text = f.read()
         train.append([text,0])

    path = os.path.dirname(__file__)+"/dataset/deportes/deportes"+str(2*i+2)+".txt"
    with open(path) as f:
         text = f.read()
         test.append([text,0])

    #Salud
    path = os.path.dirname(__file__)+"/dataset/salud/salud"+str(2*i+1)+".txt"
    with open(path) as f:
         text = f.read()
         train.append([text,0])

    path = os.path.dirname(__file__)+"/dataset/salud/salud"+str(2*i+2)+".txt"
    with open(path) as f:
         text = f.read()
         test.append([text,0])

    #politica
    path = os.path.dirname(__file__)+"/dataset/politica/politica"+str(2*i+1)+".txt"
    with open(path) as f:
         text = f.read()
         train.append([text,0])

    path = os.path.dirname(__file__)+"/dataset/politica/politica"+str(2*i+2)+".txt"
    with open(path) as f:
         text = f.read()
         test.append([text,0])
