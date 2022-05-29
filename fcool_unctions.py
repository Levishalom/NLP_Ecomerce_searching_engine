# import libs and prepare transformators for data preprossessing
import string
import pandas as pd
from string import digits
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
# import nltk
# nltk.download('omw-1.4')
# nltk.download('stopwords')
# nltk.download('punkt')

lemmatizer = WordNetLemmatizer()

stop_words = stopwords.words('english')
stop_words.extend(['manufacturer', 'cm', 'l', 'm', 'kg', 'g', 'w', 'r', 'ø',
                    'brand', 'new', 'something', 'tag', 'description', 'material', 'width', 'weight',
                    'volume', 'na', 'content', 'no', 'mm', 'piece', 'ml', 'µf', 'µh', 'µm', 'μf'])

spcial_char_map = {ord('ä'): 'a', ord('ü'): 'u', ord('ö'): 'o', ord('ß'): 's'}
# mega-function that cleans up the corpus: tokenizes, lemmatizes, removes verbs, etc.


def krasavchik(jj):

    # safe the category name
    if "'title':" in str(jj):
        # collapse to string
        jj=str(jj).replace("'children':", '').replace("'title':", '').replace("_", ' ')
    else:
        jj=str(jj).replace("_", ' ')
    # remove parentheces
    # jj=re.sub(r'\([^)]*\)', '', jj)

    # remove punctuation and to lower
    jj=jj.translate(str.maketrans('', '', string.punctuation)).lower()

    # remove numeric values
    jj = jj.translate(str.maketrans('', '', digits))

    # replace umlauds with english alternatives
    jj=jj.translate(spcial_char_map)

    # tokenize
    jj=word_tokenize(jj)

    # lemmatize
    jj=list(map(lemmatizer.lemmatize, jj))

    # REMOVE verbs
    jj=pd.DataFrame(pos_tag(jj))
    jj.columns=['words', 'part']
    jj=jj[~jj.part.isin(['VBD', 'VBN', 'VBP', 'VBZ'])]

    # remove empty values
    # jj=list(filter(('').__ne__, jj))

    # remove stopwords and duplicates (we will not account for frequencies due to nature of quering)
    jj=jj.drop_duplicates(subset=['words'])
    jj=jj[~jj.words.isin(stop_words)].words.tolist()
    return jj
