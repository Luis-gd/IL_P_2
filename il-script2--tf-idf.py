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

#Parameter to dectivate or activate prints
verbose = True
#Global Parameter: size of Glosary
top_k = 60
###############################################################################
################### PRE PROCESSING ############################################
###############################################################################

#Import stop words:
#nltk.download('stopwords')
#nltk.download('punkt')
stop_words = set(stopwords.words('spanish'))
#print(stopwords.words())

#Tokenizer
tokenizer = RegexpTokenizer(r'[A-Za-z_À-ÿ]{3,}')
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
            if len(text) < top_k:
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
             if len(text) < top_k:
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

#IDF  initialization

IDF = dict.fromkeys(union,0)

#Compute IDF values
for token in union:
    #Train docs
    for doc in train:
        if token in doc[0]:
            IDF[token] += 1
    #Test document
    for doc in test:
        if token in doc[0]:
            IDF[token] += 1

#IDF Values
for token in union:
    IDF[token] = math.log10((N/(1+IDF[token])))

#Compute tf-idf glossary of a single document with label [tokenized_text, label]
#Computes array and glossary with top_k terms
def tf_idf(doc, idf, top_k):
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

    for token in doc[0]:
        dic_doc_tf_idf[token] = (idf[token]) * (dic_doc_tf[token])

    l = sorted(dic_doc_tf_idf.items(), key=itemgetter(1), reverse=True)
    l = l[:top_k]
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

    if k < top_k:
        print("WARNING: j = "+str(j))
    return [arr, l]

#Takes pair_train_set_vectors: a pair of a vector embeding a the train set documents and it's category [vec: array, cat: int]
#and a document ([doc: str,category: int]) returns a category and vector with distance to each category
#It computes using distance "dist":
#   dist = cosine, dist = hamming, dist = q->q-norm
def categorize(pair_train_set_vectors, arr, dist):
    cat = [0]*len(categories)
    for v in pair_train_set_vectors:
        if dist == 'cosine':
            cat[v[1]] += distance.cosine(v[0],arr)
        elif dist == 'euclidean':
            cat[v[1]] += distance.euclidean(v[0],arr)
        elif dist == 'hamming':
            cat[v[1]] += distance.hamming(v[0],arr)
        elif dist == 'cityblock':
            cat[v[1]] += distance.cityblock(v[0],arr)
        elif dist == 'chebyshev':
            cat[v[1]] += distance.chebyshev(v[0],arr)

    return [np.argmin(cat), cat]

# Compute tf-idf Glosary for elements of train:
train_set_vectors_tuples = []
for doc in train:
    tuple = []
    l = tf_idf(doc,IDF,top_k)
    tuple.append(l[0])
    tuple.append(doc[1])
    train_set_vectors_tuples.append(tuple)

#Compute tf-idf for test set
correct = 0
total = 0
for doc in test:
    gl_tuple = tf_idf(doc, IDF, top_k)
    gl = gl_tuple[0]
    dist = [0] * len(categories)
    cat = categorize(train_set_vectors_tuples, gl, 'cosine')
    total += 1
    if cat[0] == doc[1]:
        correct += 1
    elif verbose:
        print(str(cat[1])+" Clasified as: "+categories[cat[0]]+", correct category is: "+categories[doc[1]])
        for tuple in gl_tuple[1]:
            print(tuple)

print("Accuracy: "+str(correct/total))
