# Author-identification, Unsupervised learning models
in this directory, you can find the baslines, and best un-supervised learning models we constructed.
Below you will find documentation on how to run each of the models and their performance

## Baseline - Dump model
* Baseline based on top of basic stylometry features as word count, charechters count, etc.
* coded in class unsupervised-pipeline-baseline.py.
to run the model, and output purity + NMI score, simply execute:
```
python unsupervised-pipeline-baseline.py
```
Performance:
* Purity = 0.137
* NMI = 0.227

## Experimental settings with sentence transformation
* experimental model that uses transformed sentence dataset, with similar set of features of our best model
can be found in class unsupervised-pipeline-sentence.py.
* to run the model, and output purity + NMI score, simply execute:
```
python unsupervised-pipeline-sentence.py
```

Performance:
* Purity = 0.113
* NMI = 0.142
* Visualizations

![](../../exps/unsupervised/2d_clustering_sentences.png?raw=true&s=100 "sentences output")
## Best Model
our best model used original version of dataset, with different set of features including: word n-grams, character n-grams, PCA.

code can be found in class unsupervised-pipeline.py.

Input params:
* --number_of_clusters=<int from 1 to 50>
* --word_ngram_dim_reduction=<int from 1 to 100>
* --draw_clustering_output=<True / False>

To run the model, print purity + NMI score, and draw clustering output in 2Dim space, simply execute:
```
python unsupervised-pipeline.py
```
Performance:
* Purity = 0.53 - 0.54
* NMI = 0.68-0.702
* Visualizations
   
![Alt text](../../exps/unsupervised/50_clustersA.JPG?raw=true "model clustering output")
