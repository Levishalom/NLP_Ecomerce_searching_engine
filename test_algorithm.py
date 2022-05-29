import pandas as pd
# import final quering algorithm
from final_match import final_quering

# for nice print
pd.set_option('display.width', 40)

# import the catalog of manufacturers
manufacturers=pd.read_csv('manufacturers.csv', index_col=None)

# test the algorithm's performance by feeding your own query


def test_algo():

    # test - write your query
    query = input('Search: ')

    # results
    predicted_categories = final_quering(query)

    if predicted_categories=='There are no good matches to your request :(\nTry rephrazing.':
        return print(predicted_categories)

    else:
        # let's print the manufacturers for the recomended products
        output = manufacturers[manufacturers.sub_title.isin(predicted_categories)][['sub_title',
                                                                                    'manufacturers']].reset_index(drop=True)

        # make sure the order is correct - according to model's predictions
        output['sort_cat'] = pd.Categorical(output['sub_title'], categories=predicted_categories, ordered=True)
        output.sort_values('sort_cat', inplace=True)
        output.reset_index(inplace=True, drop=True)
        output.drop(columns='sort_cat', inplace=True)

        return print(output)

test_algo()
