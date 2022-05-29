import json
import string
import pandas as pd
import numpy as np
import re
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import CountVectorizer
# import our tokenization, lemmatization, cleaning, etc. function
from fcool_unctions import krasavchik


# Data prep
# load data
with open('data.json', 'r') as file:
    data = file.read().replace('\xa0', ' ')

data=json.loads(data)
# basic json transformations to df

df=pd.json_normalize(data).transpose()

df=df[df.index.str.contains('children')]
df=df.reset_index()
df.columns=['big_category', 'nice']


def rep(x):
    return x.split('.')[0]
df.big_category=df.big_category.apply(rep)
# expand the jsons - one layer further
df1=pd.concat([df.big_category, pd.json_normalize(df.nice)], axis=1)

df1.index=df1.big_category
df1.drop(columns='big_category', inplace=True)


# Extract info from JSON
# our uncleaned corpus to list of jsons
listy=df1.values.reshape(-1)
listy=listy[listy != np.array(None)]
# expand the list of jsons 2 layers deeper
pre_final_corpus = dict()

for i in listy:
    yeboy=pd.json_normalize(i, 'children', meta='title', meta_prefix='ll_')
    for j in yeboy.title:
        # dict of jsons
        pre_final_corpus[yeboy[yeboy.title==j].title.values[0]] = yeboy[yeboy.title==j].values.tolist()
# list-catalouge will be used to enrich the user query results with parameters
product_info = []

# extract parameters for each (sub)category
for i in pre_final_corpus.keys():

    try:
        # we deal with the hirerachy granularity (depth)
        if 'tag' in str(pre_final_corpus[i][0]):

            objects = pre_final_corpus[i][0][3]
            bff=pd.json_normalize(pre_final_corpus[i][0][3], 'tags', meta='title', meta_prefix='sub_')

            # remove empty rows
            bff=bff[bff.children.apply(len)>0]

            # add higher level item categories
            bff['ob_class']=i

            # add to list
            product_info.append(bff)

        else:
            objects = pre_final_corpus[i][0][2]

            # extract from json
            bff=pd.json_normalize(objects)

            # remove empty rows
            bff=bff[bff.children.apply(len)>0]

            # add higher level item categories
            bff['sub_title']=i
            bff['ob_class']=i

            # add to list
            product_info.append(bff)
    except:
        # add to dict
        product_info.append(pd.DataFrame({'title': [''], 'children': [''],
                            'sub_title': [i], 'ob_class': [i]}))
    # drop the interm. df
    bff=None
# final dataframe with all the lists
product_info = pd.concat(product_info)
product_info.loc[product_info.title.apply(str).str.contains('Manufacturer'), 'title']='Manufacturer'

# concat columns (str) to construct the bag of words
product_info['corpus']=product_info.title + ' ' + product_info.children.apply(str) +\
    ' ' + product_info.sub_title + ' ' + product_info.ob_class


# NLP: data pre-processing

# find synonyms for each word in corpus (except brand names!) to enrich the corpus
#  - we have no discriptions in original set!


def synomizer(x, uni_manufacturers, max_n_synonyms=3, max_depth=4):

    # we will not try to find a syns for brand names
    if x in uni_manufacturers:
        return x
    try:
        j = wn.synsets(x)[0].lemma_names()

        if len(j)==1:
            for i in range(1, max_depth):
                j = wn.synsets(x)[i].lemma_names()
                if len(j)>1:
                    break

        # # remove words that appear in category names - since we risk bluring the categorization
        # j = [e for e in j if e not in categories]

        return krasavchik(j[0:max_n_synonyms] + [x])

    except:
        return x

# Manufacturers catalog
# get just manufacturers from the catalog
manufacturers = product_info[product_info.title.isin(['Manufacturer', None])].drop(columns='title')

# Function to convert list of words to string


def listToString(s):
    s=str(s)
    s=s.translate(str.maketrans('', '', string.punctuation))
    return s

# get list of manufacturers for each product


def transformy(jj):

    # if no info on manufacturers
    if jj is None:
        return 'no info on manufacturers'
    # collapse to string
    jj=str(jj).replace("'children':", '').replace("{'title':", '')
    # remove values between parentheces
    jj=re.sub(r'\([^)]*\)', '', jj)
    # remove punctuation
    jj=jj.translate(str.maketrans('', '', string.punctuation.replace(',', '')))

    # return list of manufacturers in original spelling
    return " ".join(jj.strip().split()).split(' , ')

# get list of manufacturers
manufacturers['manufacturers']=manufacturers.children.apply(transformy)

# cleaned manufacturers
manufacturers['cleaned']=manufacturers.children.apply(krasavchik)

# save manufacturers
manufacturers.to_csv('manufacturers.csv', index=True)

# list of unique manufacturers in our DB
uni_manufacturers = list(set(manufacturers['cleaned'].values.sum()))

# find synonyms for each word in corpus (except brand names!) to enrich the corpus
#  - we have no discriptions in original set!


# Our corpus

final=product_info[['sub_title', 'corpus']].groupby(['sub_title'], as_index=False).agg({'corpus': ' '.join})
final.corpus = final.corpus.apply(krasavchik)
final.columns=['names', 'corpus']

# final=pd.DataFrame({'names': pre_final_corpus.keys(),
#                     'corpus' : list(map(krasavchik,pre_final_corpus.items())) } )


# let's enrich our corpus with synonyms, since the items have no descriptions to be used in training
final['corpus_with_syn']=final.corpus.apply(lambda x: list(map(synomizer, x)))
# add appropriate string for dtm
final['corpus_me']=final['corpus_with_syn'].apply(listToString)

# remove limited categories
final = final[(final.corpus.map(len)>4) | (~final.names.isnull())]
# finally we are ready to train the classifier!
final[['names', 'corpus_me']].head()
# DTM - my approach: classifier trained on tf-idf bag of words + fuzzy matching
# DTM - binary + tfidf (to account for doc size), since we don't care about frequencies - no descriptions!!!

vectorizer = CountVectorizer(binary=True)
count_array = vectorizer.fit_transform(final.corpus_me.tolist()).toarray()
dtm = pd.DataFrame(data=count_array, columns=vectorizer.get_feature_names_out())

# our classes
dtm.index=final.names

# save to csv
dtm.to_csv('dtm.csv', index=True)
