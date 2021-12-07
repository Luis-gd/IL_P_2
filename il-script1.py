#Imports
import os
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from operator import itemgetter

###############################################################################
################### PRE PROCESSING ############################################
###############################################################################

#Import stop words:
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('spanish'))
#print(stopwords.words())

#Tokenizer
tokenizer = RegexpTokenizer(r'\w+')

#Read files, tokenize and remove stopwords
train = []
test = []

categories = ["deportes", "salud", "politica"]
j=0#j=0->deportes, j=1->salud, j=2->politica

for i in range(30):
    j=0#j=0->deportes, j=1->salud, j=2->politica
    for cat in categories:
        #Train set
        path = os.path.dirname(__file__)+"/dataset/"+cat+"/"+cat+str(2*i+1)+".txt"
        with open(path) as f:
            text = f.read()
            #text = text.replace('\n', ' ')
            #text_tokens = text.split(' ')
            text_tokens = tokenizer.tokenize(text)
            filtered_text = [w for w in text_tokens if not w.lower() in stop_words]
            train.append([filtered_text,j])
        #Test set
        path = os.path.dirname(__file__)+"/dataset/"+cat+"/"+cat+str(2*i+2)+".txt"
        with open(path) as f:
             text = f.read()
             #text = text.replace('\n', ' ')
             #text_tokens = text.split(' ')
             text_tokens = tokenizer.tokenize(text)
             filtered_text = [w for w in text_tokens if not w.lower() in stop_words]
             test.append([filtered_text,j])

        j+=1


###############################################################################
####################### TF-IDF ################################################
###############################################################################

#print(train[0][0])
union = []
N = len(train)
#Union set of all tokens
for doc in train:
    union = set(union).union(set(doc[0]))

#Total document frec:
frecTotal = dict.fromkeys(union,0)

#Frecuency by label, TF, IDF and TF-IDF initialization
category_union = []
TF = []
IDF = []
TF_IDF = []
for cat in categories:
    category_union.append(dict.fromkeys(union,0))
    TF.append(dict.fromkeys(union,0))
    IDF.append(dict.fromkeys(union,0))
    TF_IDF.append(dict.fromkeys(union,0))

#Frecuency of cateogries and total Frecuency
for doc in train:
    for token in doc[0]:
        category_union[doc[1]][token] += 1
        frecTotal[token] += 1

#Compute TF values for entire document
for token in union:
    for j in range((len(TF))):
        TF[j][token] = category_union[j][token]/frecTotal[token]

#Compute IDF values
for token in union:
    for doc in train:
        if token in doc[0]:
            IDF[doc[1]][token] += 1

#IDF Values and TF-IDF
for token in union:
    for j in range((len(TF))):
        IDF[j][token] = N/(1+IDF[j][token])
        TF_IDF[j][token] = TF[j][token] * IDF[j][token]

for j in range(len(categories)):
    print("------------------------------------------------------")
    print("---------"+categories[j]+"----------------------------")
    print("------------------------------------------------------")
    l = sorted(TF_IDF[j].items(), key=itemgetter(1), reverse=True)
    i = 0
    for x in l:
        print(x)
        i += 1
        #if i>100:
        #    break
