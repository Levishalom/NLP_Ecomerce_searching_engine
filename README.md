# E-comerce searching engine using NLP

This project is my take on the following challenge: https://github.com/Lhotse-Technologies/take-home-data-scientist (read the description in the repository).

The objective was to come up with an efficient recomindation algorithm that would recomend the items and the manufacturers listed in a database based on the user's unstrucutred vague description of their desired prodocut (the description is vague since they do not know the precise parameters and attributes, i.e. length, weight, material, etc. as stated in the challenge). The priorities of the project were 1. relativelly high accuracy in recomindation (that can be assessed heuristically by anyone: by typing in a query and observing how close the results of the matching are) 2. relativelly high efficiency and scalability of the algorithm (given potential growth of the database and items' list & given the possibility of large user inputs).


## Our approach:

Instead of naivelly iterating through the database, trying to (fuzzy) match each term in the user's query to the items, then somehow figuring out how to assign the weights to the predictions based on the term distances - leading to a rather ineficient algorithm with the complexity of atleast O(n^m), I have decided to combine machine learning, in particular - classifier trained on the bag-of-words, with fuzzy matching and conventional NLP preprocessing (tokenization, lemmatization, removal of verbs, etc.) that yielded rather fast, stable to the variation in useer-query length and in size of the database, accurate recomindation algorithm. 


## What has been done:

Putting aside some minor details on json parsing, let us outline major strategies implemented in the algorithm for  a. maximization of accuracy b. minimization of computation time.

1. As was already mentioned, we applied the bag-of-words strucutre, i.e. - each item was assumed to be a document that contained certain words, describing the nature of it. Obviousely, unlike human generated bodies of text, in our case the order of the words does not matter, which makes the choice of bag of words approach natural. We generated a binary DTM (document term matrix) - since the frequencies of words in our pseudo documents do not give any meaningful information. The tf-idf was tried but based on the results of the model, the factoring of document size turns out to be unnecessary for this type of problem. 
2. Before describing the training of the model let's understand how the bag of words was cleaned (done by my function krasavchik()):
        1. all words connected to the item got dumped into one list
        2. they were tokenized, lemmatized, the stopwords and verbs were removed (i.e. in user query we do not care of such words as 'i need', 'i want'...)
        3. Punctuation and numbers as well (yes, we lose some info on items but this info would be meaningless in terms of the user query, see above)
        4. some other minor cleaning has been done...

3. And bag of words generated: given the item documents lack ANY DESCRIPTIONS of the products - we had to be creative to broaden the corpa for efective training:
I have written the synomizer() function that finds the syninyms to all words in corpa (except brand names) - not to have the dtm exploded we have set the max_n_synonyms=3 for each word.


### Model training 

At this point the dtm is ready for trainig of a simple Multinomial Naive Bayes classifier (this model is nutoriuosly effective with the bag of words problems), whose sole purpose would be to output probabilities for each document given a certain user input. We also have the certanty_thresh=0.9 - which is a "credible region" of the recomindations - we say that we only want to see the recomindations that have the matching probability grater than 


### Final (fuzzy) matching and algorithm performance

All of the steps before are only done once (for every expansion of database). The last step - an algorithm that would match the user inputs with the dtm features (teerms) and output the probablistic recomidations based on the model and query. 

Firstly, we would apply the same cleaning and prep process - krasavchik() function - on user query - hence dramatically dropping the number of words, leaving only the most essential: context-creating terms. Then, we would first strictly match the terms with the dtm, and only apply the fuzzy matching to those terms that did not have any matches. We decided to apply the fuzzy matching only between the terms that start with the same letter (common practice), since people rearly make spelling mistakes on the very first letter - hence dramatically decreasing computational internsity of the final_quering() function. 

For unmatched terms we claculate the Levenshtein distance (with transposition) to the same-letter dtm terms, if the distance is > 4 (default thresh) we assume that the word from the query does not appear in our dtm. The inverse of the distance 1/distance is fed to the trained classifier - which can be interpreted as a probabilistc proxy for a query term TO really BE a term from the dtm -- this is the key to the model's accuaracy.  

Finally, we also set the certanty_thresh = 0.9, threshold for our certanty in top 1 prediction - i.e. the "credible region" to our probabilistic recomindations which we set to 90%. In other words we would output all the items ('documents') that have the probability of matching > certanty_thresh multiplied by the highest probability (i.e. the top 1 recomindation). 

If the number of recomindations exceeds the 50% of the entire pool of items (in our case there aprox. 1600 items in the DB - and for instance the query 'macbook' returns more than 800 recomindations given the certanty_thresh, the algorithm will say that there was no good match to the query).

You can set certanty_thresh to None if ypu want to load just the top 1 recomendation. Obviousle, you can also set a maximum fixed number of recomindations you want to have every time. 


## Our repository:
1. data.json - the raw json file provided in the challenge - the data
2. all_in_one.ipynb - a notebook that contains the entire flow of the algorithm (from data preps to testing) - has all the descriptions: check it out :)
3. You can also find prepaired for workflow-schedualing python scripts (each of them does a certain task - all of which can be found in the notebook):
    1. fcool_unctions.py - this is a file with the ready-to-go corpa cleaning function (krasavchik) - this file is imported in get_corpus.py and final_match.py 
    2. get_corpus.py - this script's input is the data.json, it is parsed, transformed to necessary df strucuture. the krasavchik() function is applied, synonyms generated, and the dtm constructed. The script outputs two csvs: dtm.csv and manufacturers.csv - a cleaned catalog of manufacturers that is going to be used in the final script. (neither of the csvs are loaded in this repository due to teir sizes).
    3. train_model.py - here we train the classifier based on the dtm.csv -> the pickled model mini_model.sav is the output - which is also rather havey to be added in the repository (feek free to recreate the model by running the all_in_one.ipynb).
    4. final_match.py - the final query matching function that inputs the user query - the mini_model.sav is imported for matching
    5. train_model.py - the script for testing the algorithm's performance - you can ran this in the terminal after running all the other files - it has the user input functionality
