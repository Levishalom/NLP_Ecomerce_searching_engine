# NLP E-comerce searching engine

This project is my take on the following challenge: https://github.com/Lhotse-Technologies/take-home-data-scientist (read the description in the repository).

The objective was to come up with an efficient recomindation algorithm that would recomend the items and the manufacturers listed in a database based on the user's unstrucutred vague description of their desired prodocut (the description is vague since they do not know the precise parameters and attributes, i.e. length, weight, material, etc. as stated in the challenge). The priorities of the project were 1. relativelly high accuracy in recomindation (that can be assessed heuristically by anyone: by typing in a query and observing how close the results of the matching are) 2. relativelly high efficiency and scalability of the algorithm (given potential growth of the database and items' list & given the possibility of large user inputs).

### Our approach:

Instead of naivelly iterating through the database, trying to (fuzzy) match each term in the user's query to the items, than somehow figuring out how to assign the weights to the predictions based on the term distances - leading to a rather ineficient algorithm with the complexity of atleast O(n^m), I have decided to combine machine learning, in particular - classifier trained on the bag-of-words, with fuzzy matching and conventional NLP preprocessing (tokenization, lemmatization, removal of verbs, etc.) that yielded rather fast, stable to the variation in useer-query length and in size of the database, accurate recomindation algorithm. 

#### What has been done:

Putting aside some minor details in json parsing, let us outline major strategies implemented in the algorithm to a. maximize accuracy b. minimize computation time.

1. As was already mentioned, we applied the bag-of-words strucutre, i.e. - each item was assumed to be a document that contained certain words, describing the nature of it. 

### Our repository:
1. data.json - the raw json file provided in the challenge - the data
2. all_in_one.ipynb - a notebook that contains the entire flow of the algorithm (from data preps to testing) - has all the descriptions: check it out :)
3. You can also find prepaired for workflow schedualing python scripts (each of them does a certain task - all of which can be found in the notebook):
    1. get_corpus.py - this script input is the data.json, it is parsed, transformed to necessary df strucuture, which is than 

