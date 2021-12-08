#Imports
import os
import nltk
import math
import numpy as np

from scipy.special import logsumexp
from scipy.spatial import distance
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from operator import itemgetter

###############################################################################
################### PRE PROCESSING ############################################
###############################################################################

#Import stop words:
#nltk.download('stopwords')
#nltk.download('punkt')
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
            if len(text) < 100:
                print("# WARNING: text "+categories[j]+str(2*i+1)+".txt, might be empty ")
                print("text:\n"+text)
            #text = text.replace('\n', ' ')
            #text_tokens = text.split(' ')
            text_tokens = tokenizer.tokenize(text)
            filtered_text = [w for w in text_tokens if not w.lower() in stop_words]
            train.append([filtered_text,j])
        #Test set
        path = os.path.dirname(__file__)+"/dataset/"+cat+"/"+cat+str(2*i+2)+".txt"
        with open(path) as f:
             text = f.read()
             if len(text) < 100:
                 print("# WARNING: text "+categories[j]+str(2*i+1)+".txt, might be empty ")
                 print("text:\n"+text)
             #text = text.replace('\n', ' ')
             #text_tokens = text.split(' ')
             text_tokens = tokenizer.tokenize(text)
             filtered_text = [w for w in text_tokens if not w.lower() in stop_words]
             test.append([filtered_text,j])

        j+=1


###############################################################################
####################### TF-IDF ################################################
###############################################################################

#Compute TF-IDF glossary of the entire training set
#print(train[0][0])
union = []
N = len(train)+len(test)
#Union set of all tokens
for doc in train:
    union = set(union).union(set(doc[0]))

#union_train = union

for doc in test:
    union = set(union).union(set(doc[0]))

#Total document frec:
frecTotal = dict.fromkeys(union,0)

#Frecuency by label, TF, IDF and TF-IDF initialization
category_union = []
TF = []
IDF = dict.fromkeys(union,0)
TF_IDF = []
for cat in categories:
    category_union.append(dict.fromkeys(union,0))
    TF.append(dict.fromkeys(union,0))
    #IDF.append(dict.fromkeys(union,0))
    TF_IDF.append(dict.fromkeys(union,0))

#Frecuency of cateogries and total Frecuency
for doc in train:
    for token in doc[0]:
        category_union[doc[1]][token] += 1
        frecTotal[token] += 1

#Compute TF values for entire document
for token in union:
    for j in range((len(TF))):
        TF[j][token] = category_union[j][token]#/frecTotal[token]

#Compute IDF values
for token in union:
    for doc in train:
        if token in doc[0]:
            IDF[token] += 1

    for doc in test:
        if token in doc[0]:
            IDF[token] += 1

#IDF Values and TF-IDF
for token in union:
    IDF[token] = math.log10((N/(1+IDF[token])))
    for j in range((len(TF))):
        TF_IDF[j][token] = (TF[j][token]) * (IDF[token])

## Get 65 Top elements as a glossary, Print
glossary = [[]] * len(categories)
glossary_arr = [0] * len(categories)
for j in range(len(categories)):
    glossary[j] = []
    glossary_arr[j] = []
    print("------------------------------------------------------")
    print("---------"+categories[j]+"--------------------------------")
    print("------------------------------------------------------")
    l = sorted(TF_IDF[j].items(), key=itemgetter(1), reverse=True)
    glossary[j] = l[:65]
    k=0
    for token in union:
        match = False
        for tuple in glossary[j]:
            if token == tuple[0]:
                match = True
                break
        if match:
            glossary_arr[j].append(1)
            k += 1
        else:
            glossary_arr[j].append(0)

    if k < 65:
        print("WARNING: j = "+str(k))

    #print(glossary_arr[j])
    str_print = str(glossary[j]).strip('[]')
    #print("Size: "+str(len(glossary[j])))
    #print("Glosary ("+str(j+1)+"):\n"+str_print.replace('),',')\n'))

#Compute tf-idf glossary of a single document with label [tokenized_text, label]
def tf_idf(doc, idf):
    dic_doc_tf = dict.fromkeys(union,0)
    dic_doc_tf_idf = dict.fromkeys(union,0)
    #TF
    all_zero_tf = True
    for token in doc[0]:
        dic_doc_tf[token] += 1
        if dic_doc_tf[token] > 0:
            all_zero_tf = False

    if(all_zero_tf):
        print("# WARNING: All zero tf")
        print("Category "+categories[doc[1]]+" Doc:\n"+str(doc[0]))

    all_zero_idf = True

    for token in doc[0]:
        dic_doc_tf_idf[token] = (idf[token]) * (dic_doc_tf[token])
        if idf[token] > 0:
            all_zero_idf = False

    if(all_zero_idf):
        print("# WARNING: All zero idf")

    l = sorted(dic_doc_tf_idf.items(), key=itemgetter(1), reverse=True)
    l = l[:65]
    #Compute array
    arr = []
    k=0
    for token in union:
        match = False
        for tuple in l:
            if token == tuple[0]:
                match = True
                break
        if match:
            arr.append(1)
            k += 1
        else:
            arr.append(0)

    if k < 65:
        print("WARNING: j = "+str(j))
    return [arr, l]

#Compute tf-idf for test set
correct = 0
total = 0
for doc in test:
    gl_tuple = tf_idf(doc, IDF)
    gl = gl_tuple[0]
    dist = [0] * len(categories)
    for j in range(len(categories)):
        if(len(gl) == len(glossary_arr[j])):
            dist[j] = ((np.dot(gl,glossary_arr[j])))/np.sqrt((np.dot((gl),(gl)))*(np.dot((glossary_arr[j]),(glossary_arr[j]))))
            if math.isnan(dist[j]):
                print("NaN Error ("+str(j)+").")
                print("gl: "+str(gl))
                print("glossary_arr[j]: "+str(glossary_arr[j]))
        else:
            print("Error, size of gl = "+str(len(gl))+", and size of glossary_arr = "+str(len(glossary_arr[j])))
        #print(str(gl)+' * '+ str(glossary_arr[j]) + " = " + str(dist[j]))
    if max(dist) < 0.00000001:
        print("WARNING: Zero distance vector cause by this glossary:")
        #for tuple in gl_tuple[1]:
        #    print(tuple)
    cat = np.argmax(dist)
    #print(str(dist)+" Clasified as: "+categories[cat]+", correct categorie is: "+categories[doc[1]])
    total += 1
    if cat == doc[1]:
        correct += 1
    else:
        print(str(dist)+" Clasified as: "+categories[cat]+", correct category is: "+categories[doc[1]])

print("Accuracy: "+str(correct/total))
