import pandas as pd
import pickle

dtm=pd.read_csv('dtm.csv', index_col=0)

# Train simple Multinomial Naive Bayes classifier - no testing =(
# our model
from sklearn.naive_bayes import MultinomialNB
mini_model = MultinomialNB()
mini_model.fit(dtm, dtm.index)

# save our model
pickle.dump(mini_model, open('mini_model.sav', 'wb'))
