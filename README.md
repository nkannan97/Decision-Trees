Decision Tree

implemented an ID3-like decision-tree learner for binary classification. The program reads files in ARFF format, the feature values are   separated by commas, and the last value on each line is the class label of the instance. Each ARFF file starts with a header section describing the features and the class labels. Lines starting with '%' are comment.

The program can be run through the command line through the following command line arguments
"dt-learn train-set-file test-set-file m"
where m is the number of data instances per node
  

