#Imports
import os
import nltk
import math
import numpy as np
import time

from scipy.special import logsumexp
from scipy.spatial import distance
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from operator import itemgetter

###############################################################################
################### PRE PROCESSING ############################################
###############################################################################
def read(categories, top_k):
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
                     print("# WARNING: text "+categories[j]+str(2*i+2)+".txt, might be empty ")
                     print("text:\n"+text)
                 #text = text.replace('\n', ' ')
                 #text_tokens = text.split(' ')
                 text_tokens = tokenizer.tokenize(text)
                 filtered_text = [w for w in text_tokens if not w.lower() in stop_words]
                 test.append([filtered_text,j])

            j+=1

    return [train, test]


###############################################################################
####################### TF-IDF ################################################
###############################################################################

#Compute IDF glossary of the entire training set
#print(train[0][0])
def idf(train, test):
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


    return [IDF, union]
#return [glossary_arr, glossary]
def category_glossary(top_k, verbose, union, IDF, train, categories):
    category_union = []
    TF = []
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

    #Compute TF values for entire document
    for token in union:
        for j in range((len(TF))):
            TF[j][token] = category_union[j][token]#/frecTotal[token]

    #IDF Values and TF-IDF
    for token in union:
        for j in range((len(TF))):
            TF_IDF[j][token] = (TF[j][token]) * (IDF[token])

    glossary = [[]] * len(categories)
    glossary_arr = [0] * len(categories)
    for j in range(len(categories)):
        glossary[j] = []
        glossary_arr[j] = []
        if verbose:
            print("------------------------------------------------------")
            print("---------"+categories[j]+"--------------------------------")
            print("------------------------------------------------------")

        l = sorted(TF_IDF[j].items(), key=itemgetter(1), reverse=True)
        glossary[j] = l[:top_k]
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

        if k < top_k:
            print("WARNING: j = "+str(k))

        if verbose:
            str_print = str(glossary[j]).strip('[]')
            print("Size: "+str(len(glossary[j])))
            print("Glosary ("+str(j+1)+"):\n"+str_print.replace('),',')\n'))

    return [glossary_arr, glossary]


#Compute tf-idf glossary of a single document with label [tokenized_text, label]
#Computes array and glossary with top_k terms
def tf_idf(doc, idf, top_k, categories, union):
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
def categorize(v_cat, v2, dist, categories):
    cat = [0]*len(categories)
    for i in range(len(categories)):
        if dist == 'cosine':
            cat[i] = distance.cosine(v_cat[i],v2)
        elif dist == 'euclidean':
            cat[i] = distance.euclidean(v_cat[i],v2)
        elif dist == 'hamming':
            cat[i] = distance.hamming(v_cat[i],v2)
        elif dist == 'cityblock':
            cat[i] = distance.cityblock(v_cat[i],v2)
        elif dist == 'chebyshev':
            cat[i] = distance.chebyshev(v_cat[i],v2)

    return [np.argmin(cat), cat]

# Compute tf-idf Glosary for elements of train:
def glossary_train(train,IDF,top_k,categories,union):
    train_set_vectors_tuples = []
    for doc in train:
        tuple = []
        l = tf_idf(doc,IDF,top_k,categories,union)
        tuple.append(l[0])
        tuple.append(doc[1])
        train_set_vectors_tuples.append(tuple)

    return train_set_vectors_tuples;

#Compute tf-idf for test set
def clasiffy(test, IDF, union, category_glossary, top_k, distance, categories, verbose, verbose2):

    correct = 0
    total = 0
    for doc in test:
        gl_tuple = tf_idf(doc, IDF, top_k, categories, union)
        gl = gl_tuple[0]
        dist = [0] * len(categories)
        cat = categorize(category_glossary, gl, distance, categories)
        total += 1
        if cat[0] == doc[1]:
            correct += 1
            if verbose2:
                print(str(cat[1])+" Clasified as: "+categories[cat[0]]+", correct category is: "+categories[doc[1]])
                for tuple in gl_tuple[1]:
                    print(tuple)
        elif verbose:
            print(str(cat[1])+" Clasified as: "+categories[cat[0]]+", correct category is: "+categories[doc[1]])
            for tuple in gl_tuple[1]:
                print(tuple)

    return (correct/total)

#Parameter to dectivate or activate prints: verbose
#Global Parameter: size of Glosary top_k

def run_tf_idf(top_k, distance, verbose, verbose2):
    t0 = time.time()
    categories = ["deportes", "salud", "politica"]
    data = read(categories,top_k);
    idf_union = idf(data[0],data[1])
    cg = category_glossary(top_k, verbose, idf_union[1], idf_union[0],data[0],categories)
    #train_set_vectors_tuples = glossary_train(data[0], idf_union[0],top_k,categories, idf_union[1])
    acc = clasiffy(data[1], idf_union[0], idf_union[1], cg[0], top_k, distance, categories, verbose, verbose2)
    t1 = time.time()
    print("Accuracy: "+str(acc))
    return [acc, (t1-t0)]
