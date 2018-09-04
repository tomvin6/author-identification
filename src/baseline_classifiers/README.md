# Author-identification, suppervised learning models
in this directory, you can find the baslines, tested features and suppervised learning models we constructed.
Below you will find documentation on how to run each of the models and what is their performance.

input params:
* --file=<path_to_train_data_file>
* --preprocess=<one of POS, ENT,CLN>

## Test word count features
Classifiers based on MultinomialNB on top of "Bag of words" vector (CountVectorizer), with 3-grams.
coded in class word_count.py

### Baseline - Original text, MultinomialNB
to re-train the classifier, and output log-loss_accuracy simply run 
```
python word_count.py
```
performance:
* log-loss = 1.64
* Accuracy = 0.69

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
settings of the model are as follows:
TBD

### Original text
to re-train the classifier, and output log-loss+accuracy simply run
```
python fasttext.py
```
performance:
* log-loss = 1.722
* Accuracy = 0.514

### POS teggad text
performance:
* log-loss = 
* Accuracy = 

### Entity teggad text
performance:
* log-loss = 
* Accuracy = 

### Cleaned text
performance:
* log-loss = 
* Accuracy = 

