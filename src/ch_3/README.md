# Chapter 3
## Classification
Find notebook for this chapter [here](https://github.com/hiyaryan/hands-on-ml/blob/main/src/ch_3/Classification.ipynb).

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

##### Models
- [SGDClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html) - Linear classifiers (SVM, logistic regression, a.o.) with SGD training.

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