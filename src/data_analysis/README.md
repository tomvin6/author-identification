# Author identification- Data analysis

this file is a documentation for content included in data analysis section

## Statistics.py

This class is the main program to produce data visualization.
possible input args is a path to external data.
any input data should be in the format of delimited csv file, with header, incuding columns 'text' and 'author'.
to run the program on out project data simply run "python statistics.py"

### Directory output_charts

will contain the output produced by statistics.py 
currently contain our project output in a dedicated folder.

### Directory basline_manual_analysis 

incude files used for suppervised baseline evaluation.
* baseline_evaluation_classes.tsv- file includes the validation set (sentences), with their original and baseline- predicted labels.
* baseline_evaluation_classes_errors.tsv- subset of baseline_evaluation_classes, where predicated label is defferent from true label.
* baseline_labels_encoding.csv- mapping between author names to their label encoding by baseline model.
* Confusion.pdf- visualization of confusion matrix of baseline model

