# Chapter 3
## Classification
Find notebook for this chapter [here](https://github.com/hiyaryan/hands-on-ml/blob/main/src/ch_3/Classification.ipynb).

*If someone says, "Let's reach 99% precision," you should ask, "At what recall?"*

Contents
1. [MNIST](#mnist)
2. [Training a Binary Classifier](#training-a-binary-classifier)
3. [Performance Measures](#performance-measures)
4. [Multiclass Classification](#multiclass-classification)
5. [Error Analysis](#error-analysis)
6. [Multilabel Classification](#multilabel-classification)
7. [Multioutput Classification](#multioutput-classification)

---
### Packages
- [numpy](http://www.numpy.org/)
- [matplotlib](https://matplotlib.org/)
- [sklearn](http://scikit-learn.org/stable/)

#### matplotlib
- [pyplot](https://matplotlib.org/api/pyplot_api.html) - provides a MATLAB-like plotting framework

#### sklearn
##### Data
- [datasets](http://scikit-learn.org/stable/datasets/index.html) - helper functions to download popular datasets
  - [fetch_openml](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_openml.html) - Fetch dataset from [openml](https://www.openml.org/) by name or dataset id.
- [model_selection](http://scikit-learn.org/stable/model_selection.html) - tools for model selection and evaluation
  - [StratifiedKFold](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html) - Stratified K-Folds cross-validator (provides train/test indices to split data in train/test sets).
  - [cross_val_score](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html) - Evaluate a score by cross-validation.
  - [cross_val_predict](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html) - Generate cross-validated estimates for each input data point.
- [base](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.base) - base classes and utility functions
  - [clone](http://scikit-learn.org/stable/modules/generated/sklearn.base.clone.html) - Constructs a new estimator with the same parameters (deep copy).
- [StandardScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) - Standardize features by removing the mean and scaling to unit variance.

##### Models
- [SGDClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html) - Linear classifiers (SVM, logistic regression, a.o.) with SGD training.
- [RandomForestClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) - A random forest classifier.
- [SVC](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) - C-Support Vector Classification (SVC).
- [OneVsOneClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsOneClassifier.html) - One-vs-one multiclass strategy.
- [OneVsRestClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html) - One-vs-the-rest (OvR) multiclass/multilabel strategy.
- [metrics](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics) - for evaluating models
  - [confusion_matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html) - Compute confusion matrix to evaluate the accuracy of a classification.
  - [precision_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html) - Compute the precision (positive predictive value) metric.
  - [recall_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html) - Compute the recall (sensitivity or true positive rate) metric.
  - [f1_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html) - Compute the F1 score, also known as balanced F-score or F-measure.
  - [precision_recall_curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html) - Compute precision-recall pairs for different probability thresholds.
  - [roc_curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html) - Compute Receiver operating characteristic (ROC).
  - [roc_auc_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html) - Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.

---
---
### MNIST
[MNIST](https://en.wikipedia.org/wiki/MNIST_database) is the "hello world" of machine learning. It is a dataset of 70,000 small handwritten images of digits. Each image is labeled with the digit it represents. New classification algorithms are often tested on MNIST.

Scikit-Learn provides functions to download popular datasets like MNIST from its `sklearn.datasets`, `fetch_openml()` function.

Scikit-Learn dataset dictionary structure:
```python
{
    'DESCR': 'some description',
    'data': 'a numpy array of shape (n_samples, n_features)',
    'target': 'a numpy array of shape (n_samples,)',
    'feature_names': 'a list of length n_features',
    'target_names': 'a list of length n_classes'
}
```

A single image is represented as a 1D array of 784 features (28x28 pixels). Each feature represents one pixel's intensity, from 0 (white) to 255 (black). The `imshow()` function displays an image from an array.

To view the image, we need to reshape the array to 28x28 pixels. The `reshape()` function returns a new array with the specified shape.

MNIST dataset is already split into a training set (the first 60,000 images) and a test set (the last 10,000 images) and is already shuffled.

---
### Training a Binary Classifier
A binary classifier is a classifier that can distinguish between two classes.

A good classifier to start with is the [Stochastic Gradient Descent (SGD)](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html) classifier.
- SGD handles large datasets efficiently.
- SGD deals with training instances independently, one at a time making it well suited for online learning.

SGD relies on randomness during training. For reproducibility, set the `random_state` parameter.

---
### Performance Measures
Contents
- [Measuring Accuracy Using Cross-Validation](#measuring-accuracy-using-cross-validation)
- [Confusion Matrix](#confusion-matrix)
- [Precision and Recall](#precision-and-recall)
- [Precision/Recall Tradeoff](#precisionrecall-tradeoff)
- [The ROC Curve](#the-roc-curve)

#### Measuring Accuracy Using Cross-Validation
##### def. K-fold Cross-Validation - the training set is split into k-folds, the model is trained against k-1 folds and validated against the remaining fold, repeated k times, and reported as the average of the values computed in the loop

Accuracy is generally not the preferred performance measure for classifiers, especially when dealing with skewed datasets. A classifier that always predicts the most frequent class will have a high accuracy. If we have a dataset of 90% class A and 10% class B, a classifier that always predicts class A will have a 90% accuracy.

##### def. skewed dataset - when some classes are much more frequent than others in a dataset

#### Confusion Matrix
##### def. confusion matrix - evaluates the quality of a classification model by comparing the actual and predicted classes

Computing the confusion matrix requires a set of predictions so they can be compared to the actual targets. To get the predictions, use the `cross_val_predict()` function that ensures clean predictions are made using the training set (no data leakage).

`cross_val_predict()` performs K-fold cross-validation by splitting the dataset into K-folds, training the model on each fold, and making predictions on each fold using the model trained on that fold.

The `confusion_matrix()` function returns an array representing the confusion matrix. The rows represent the actual classes and the columns represent the predicted classes.
- Correct classifications are called true positives (TP) and true negatives (TN) and are on the main diagonal of the matrix.
- Incorrect classifications are called false positives (FP) and false negatives (FN) and are off the main diagonal of the matrix.

A perfect classifier would have only true positives and true negatives, so its confusion matrix would have nonzero values only on its main diagonal and zeros everywhere else.

##### def. precision - the accuracy of the positive predictions
TP is the number of true positives and FP is the number of false positives.
$$precision = \frac{TP}{TP + FP}$$

##### def. recall - the ratio of positive instances that are correctly detected by the classifier
TP is the number of true positives and FN is the number of false negatives.
$$recall = \frac{TP}{TP + FN}$$

#### Precision and Recall
To calculate the precision and recall, use the `precision_score()` and `recall_score()` functions from Scikit-Learn.

An $F_1$ score is the harmonic mean of precision and recall. The harmonic mean gives more weight to low values. A classifier will only get a high $F_1$ score if both recall and precision are high.

$$F_1 = \frac{2}{\frac{1}{precision} + \frac{1}{recall}} = 2 \times \frac{precision \times recall}{precision + recall} = \frac{TP}{TP + \frac{FN + FP}{2}}$$

$F_1$ score favors classifiers that have similar precision and recall. However, in some contexts, precision is more important and in other contexts, recall is more important.
##### def. precision/recall tradeoff - increasing precision reduces recall and reducing precision increases recall

#### Precision/Recall Tradeoff
The `SGDClassifier` computes a score based on a decision function. Given some decision threshold, greater scores assign the instance to the positive class, otherwise, to the negative class. 

If the threshold is half way between the scores of two instances, the classifier will be unsure about which class to assign to the instance. 
- Increasing the threshold increases precision and reduces recall. 
- Decreasing the threshold increases recall and reduces precision.

SciKit-Learn has a `decision_function()` method that returns a score for each instance. Note that `SGDClassifier` has a threshold of 0 by default.

To decide on a threshold, use the `precision_recall_curve()` function. It returns the precision and recall for all possible thresholds.

Another method to select a good precision/recall tradeoff is to plot precision directly against recall.
- With this method, select the threshold right before the precision starts to fall sharply.

A precision classifier is not very useful if its recall is too low. Setting the decision threshold to a high value increases precision but reduces recall.

#### The ROC Curve
##### def. receiver operating characteristic (ROC) curve - plots the true positive rate (recall) against the false positive rate (FPR) at various threshold settings i.e. plots the sensitivity (recall) against 1 - specificity

##### def. true positive rate (TPR) - another name for recall or the sensitivity of the classifier
$$TPR = \frac{TP}{TP + FN}$$

##### def. false positive rate (FPR) - ratio of negative instances that are incorrectly classified as positive
$$FPR = \frac{FP}{FP + TN}$$

##### def. specificity - the ratio of negative instances that are correctly classified as negative
$$specificity = \frac{TN}{TN + FP}$$

##### def. sensitivity - another name for recall
$$sensitivity = \frac{TP}{TP + FN}$$

Plot the ROC curve using the SciKit-Learn `roc_curve()` function. It returns the FPR and TPR for various threshold values.

The plot shows that the classifier produces a high recall as the threshold is increased. However, the classifier also produces a high false positive rate. A good classifier stays as far away from the dashed line as possible (toward the top-left corner).

The ROC curve is a useful tool to compare classifiers. These comparisons can be made by measuring the area under the curve (AUC). A perfect classifier will have an AUC of 1 and a purely random classifier will have an AUC of 0.5.

##### def. area under the curve (AUC) - a measure of the performance of a binary classifier

### Multiclass Classification
Multiclass classification algorithms are classifiers (also called multinomial classifiers) that can distinguish between more than two classes.

Algorithms capable of handling multiclass classification natively:
- Stochastic Gradient Descent (SGD) - classifies by assigning each class a weight vector and using the score of the class with the highest weight vector as the prediction
- Random Forest - classifies by averaging the predictions of each decision tree in the forest
- Naive Bayes - classifies by estimating the probability that an instance belongs to each class and predicting the class with the highest probability

Strictly binary classifiers:
- Logistic Regression
- Support Vector Machine (SVM)

Certain strategies can be used to perform multiclass classification using strictly binary classifiers.

One method to classify all 10 digits in the MNIST dataset is to train 10 binary classifiers, one for each digit. This would result in 10 binary classifiers for each digit (0-detector, 1-detector, etc.).

##### def. one-versus-all (OvA) or one-versus-the-rest (OvR) - a strategy to classify multiclass classification problems using strictly binary classifiers

##### def. one-versus-one (OvO) - a strategy to classify multiclass classification problems by training a binary classifier for every pair of digits

For a OvO strategy, if there are N classes, train N x (N - 1) / 2 classifiers. For the MNIST dataset, this would result in 45 binary classifiers. The advantage of OvO is that each classifier only needs to be trained on the part of the training set for the two classes that it must distinguish.

For algorithms that scale poorly with the size of the training set, OvO is preferred because it is faster to train many classifiers on small training sets than to train few classifiers on large training sets.

For algorithms that scale well with the size of the training set, OvR is preferred because it is faster to train a few classifiers on large training sets than to train many classifiers on small training sets.

SciKit-Learn can detect when a binary classification algorithm is used for multiclass classification and automatically run OvR or OvO depending on the algorithm.

To force SciKit-Learn to use a specific strategy, use the `OneVsOneClassifier` or `OneVsRestClassifier` classes and pass a binary classifier to their constructor.

### Error Analysis
To analyze errors of a promising classification model start by looking at the confusion matrix. The `matshow()` function from Matplotlib can give a visual representation of the confusion matrix.

If most images are on the main diagonal, the model classified them correctly. If there are darker squares on the main diagonal then, these squares could mean there were fewer images of that class in the dataset or the classifier does not perform well on those classes.

For he MNIST dataset, the classifier can be improve by reducing false 8s. This cn be be done by training the classifier with more images of 8s, engineering new features that would help the classifier, or preprocessing the images to make some patterns stand out more.

Use Matploblib's `imshow()` to analyze individual errors that can help you gain insight into what the classifier is doing and why it is failing which will help you choose the ways to improve the classifier.

The classifier gets confused between 3s and 5s. Some of these are due to the way the digits are poorly written, but others are due to the classifier being too simple. SGD is a linear model that assigns a weight per class to each pixel. When the SDG model sees a new image, it sums up the weighted pixel intensities to get a score for each class. 3s and 5s differ only by a few pixels, so the model easily confuses them. This classifier is sensitive to image shifting and rotation, so preprocessing the images to ensure they are well centered and not too rotated will help reduce the number of false 3s and 5s.

### Multilabel Classification
Multilabel classification is when a classifier outputs multiple binary tags. For example, a face recognition classifier might output multiple people in the same picture.

One approach to selecting a multilabel classifier is to measure the F1 score for each individual label and then compute the average score.

Not all labels are equally important so more weight can be given to some labels to improve the classifier. This can be done by giving each label a weight equal to its support by setting `average="weighted"` in the `f1_score()` function.

##### def. support - the number of instances with that target label

### Multioutput Classification
Multioutput classification is a generalization of multilabel classification where each label can be multiclass (i.e. it can have more than two possible values).

A system that takes an image as input and outputs a digit image represented as an array of pixel intensities is an example of a multioutput classification system.