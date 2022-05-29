import pandas as pd
import numpy as np
import pickle
from nltk.metrics.distance import edit_distance
# import our tokenization, lemmatization, cleaning, etc. function
from fcool_unctions import krasavchik

# load the classifier
mini_model = pickle.load(open('mini_model.sav', 'rb'))

# pridicts the product category based on query


def final_quering(
        query,
        distince_thresh=4,  # threshold for levinshtein distance
        matching_by_first_letter=True,  # apply fuzzy matching only for the words that start from the same letter - major efficency boost
        certanty_thresh=0.9,  # threshold for our certanty in top 1 prediction - i.e. "credible region" we set to 95% from top proba
        # (set certanty_thresh to None if ypu want to load the top 1 recomendation)
        max_printed_values=10,  # max number of recomanded items
        show_more=False  # show more info on the results
        ):
         
    # if empty string is passed
    if query=='':
        return ''

    # clean up the query
    needed=krasavchik(query)

    # strict match
    all_features=pd.DataFrame({'words': mini_model.feature_names_in_})
    we_have_it=all_features[all_features.words.isin(needed)].words.tolist()

    # list of words that dont have a match
    unmatched = list(set(needed)-set(we_have_it))
   
    # final query dtm
    final_dataframe=pd.DataFrame(0, index=np.arange(1), columns=mini_model.feature_names_in_)
   
    # fuzzy match: I need to vectorize that search - may be later...

    if len(unmatched)>0:
       
        for i in unmatched:

            # list to store the distances at each iteration
            levin=[]

            # match only the one that starts with the same letter (people dont mistake their first letter)
            if matching_by_first_letter is True:
                # for faster perfomance
                iteration=final_dataframe.columns[final_dataframe.columns.astype(str).str[0]==i[0]].tolist()
            else:
                iteration=final_dataframe.columns.tolist()

            # let's iterate through all the words in the cleaned query    
            for j in iteration:
                dis=edit_distance(j, i, transpositions=True)
                if dis>=distince_thresh:
                    dis=0
                else:
                    # inverse of the distance - correct direction of the value
                    dis=1/dis

                levin.append(dis)
                
            final_dataframe[iteration]=levin

    # plug in the ones we have - ths the dtm for the user query
    final_dataframe[we_have_it]=1
   
    # some info on fuzzy matching
    if show_more is True:

        print('Words with perfect matching: ', we_have_it)
        print('Words with no matching - fuzzy matching applied: ', unmatched)
        # print(final_predictions.sort_values(by='proba', ascending=False).head(10))
                                   
    # we can output top 1 prediction on user query 
    if certanty_thresh is None:
       
        return mini_model.predict(final_dataframe)[0]

    # output all categories that fall under our "credible region" - recommending the best matches
    else:
        # df with  final predictions
        final_predictions=pd.DataFrame({'products': mini_model.classes_.reshape(-1), 
                                        'proba': mini_model.predict_proba(final_dataframe).reshape(-1)})
        # Credible region
        proba_thresh = (final_predictions.proba.max())*certanty_thresh

        # final list of matching products
        at_last = final_predictions[final_predictions.proba>proba_thresh]\
                                            .sort_values('proba', ascending=False).products.tolist()

        # if there is no good match (give the thresh the output = 50% all products in DB)
        if len(at_last) >= round(len(final_predictions.products)/2):
            return '''There are no good matches to your request :(\nTry rephrazing.'''

        else:
            return at_last[0:max_printed_values]
