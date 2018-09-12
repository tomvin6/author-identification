# Author-identification, suppervised learning models
in this directory, you can find the baslines, tested features and suppervised learning models we constructed.
Below you will find documentation on how to run each of the models and what is their performance.

input params:
* --file=<path_to_train_data_file>
* --preprocess=<one of POS, ENT,CLN>
* --ngram=<int values between 1 to 3>

## Baseline - Dump model
Classifier based on Logistic regression on top of basic stylometry features as word count, charechters count, etc.
coded in class suppervised_dumb_baseline.py
to re-train the classifier, and output log-loss + accuracy simply run 
```
python suppervised_dumb_baseline.py
```
performance:
* log-loss = 3.78
* Accuracy = 0.055

## Test word count features
Classifiers based on MultinomialNB on top of "Bag of words" vector (CountVectorizer), with 3-grams.
code in class word_count.py.
To configure ngram value, use input arg "--ngram=<ngram int value>". default set to 1.

### Original text, MultinomialNB
to re-train the classifier, and output log-loss_accuracy simply run 
```
python word_count.py --ngram=1 
```
performance:
* log-loss = 1.64
* Accuracy = 0.69

```
python word_count.py --ngram=3 
```
performance:
* log-loss = 2.74
* Accuracy = 0.73

```
python word_count.py --ngram=3 --preprocessing=CLN
```
performance:
* log-loss = 1.913
* Accuracy = 0.75

To test performance of classifier on external data, run 
```
python word_count.py --file=<path_to_train_data_file>
```
To test performance of classifier on external data and save plots run 
```
python word_count.py --file=<path_to_train_data_file> --plots=True
```

Note: input data should be tab-delimited, with header, including columns 'text' and 'author_label'(int).

### POS tagged text, MultinomialNB
```
python word_count.py --file=<path_to_train_data_file> --preprocess=POS
```

performance:
* log-loss = 1.755 
* Accuracy = 0.69

### Entity tagged text, MultinomialNB
```
python word_count.py --file=<path_to_train_data_file> --preprocess=ENT
```
performance:
* log-loss = 2.257 
* Accuracy = 0.55

## Test TF-IDF features
Classifiers based on MultinomialNB/Logistic Regression on top of TF-IDF features (TfidfVectorizer), with N-grams.

### Original text, MultinomialNB
to re-train the classifier, and output log-loss_accuracy simply run
```
python nb_tf_idf.py
```
performance:
* log-loss = 2.152 
* Accuracy = 0.72

To test performance of classifier on external data, run 
```
python nb_tf_idf.py --file=<path_to_train_data_file>
```
Note: input data should be tab-delimited, with header, including columns 'text' and 'author_label'(int).

### Cleaned text, MultinomialNB
stop words and panctuations annotated.
performance:
* log-loss = 1.522 
* Accuracy = 0.7

To test performance of classifier on external data, run 
```
python nb_tf_idf.py --preprocess=CLN --file=<path_to_train_data_file>
```
Note: input data should be tab-delimited, with header, including columns 'text' and 'author_label'(int)

### POS tagged text, MultinomialNB
to re-train the classifier, and output log-loss_accuracy simply run
```
python nb_tf_idf.py --preprocess=POS
```
performance:
* log-loss = 2.405 
* Accuracy = 0.68

To test performance of classifier on external data, run 
```
python nb_tf_idf.py --preprocess=POS --file=<path_to_train_data_file>
```
Note: input data should be tab-delimited, with header, including columns 'text' and 'author_label'(int).

### Entity tagged text, MultinomialNB
to re-train the classifier, and output log-loss_accuracy simply run
```
python nb_tf_idf.py --preprocess=ENT
```
performance:
* log-loss = 2.619 
* Accuracy = 0.596

To test performance of classifier on external data, run 
```
python nb_tf_idf.py --preprocess=ENT --file=<path_to_train_data_file>
```
Note: input data should be tab-delimited, with header, including columns 'text' and 'author_label'(int).

### Original text, Logistic Regression
replace "nb_tf_idf.py" with "lgr_tf_idf.py" in the above command lines.
performance:
* log-loss = 1.941  
* Accuracy = 0.73

### POS tagged text, Logistic Regression
replace "nb_tf_idf.py" with "lgr_tf_idf.py" in the above command lines.
performance:
* log-loss = 2.240  
* Accuracy = 0.72

### Entity tagged text, Logistic Regression
replace "nb_tf_idf.py" with "lgr_tf_idf.py" in the above command lines.
performance:
* log-loss = 2.400  
* Accuracy = 0.6

###  Original text,SVM
replace to "svm_tfidf.py" in the above command lines.
performance:
* log-loss = 1.722
* Accuracy = 0.514

# Test fast-text features
To implement fast-text model we used Kares package.
Classifiers based on MultinomialNB/Logistic Regression on top of TF-IDF features (TfidfVectorizer), with N-grams.

### Original text
to re-train the classifier, and output log-loss+accuracy simply run
```
python fasttext.py
```
performance:
* log-loss = 1.54
* Accuracy = 0.65

### POS teggad text
performance:
* log-loss = 1.56
* Accuracy = 0.65

### Cleaned text
performance:
* log-loss = 
* Accuracy = 

## GBM, stacked model
Params:
* --file=<path to external data file>
* --train=<True/False> (will be False if one want to run only classification of the recieved model)
* --preprocess=<True/False>
* --output_path=<path to save output file in case there are any>

to train all relevant models:
```
python xgboost_stacked_model.py --preprocess=False --train=True
```
trained model files will be under src/baseline_classifiers/xgboost_stacked_sub_mod_dumps

to classify data according to pre-trained models:
```
python xgboost_stacked_model.py --preprocess=True --train=False --file=<my_data_file>
```
output file with all features probabilities will be in provided output path.


performance:
* log-loss = 0.76
* Accuracy = 0.795
