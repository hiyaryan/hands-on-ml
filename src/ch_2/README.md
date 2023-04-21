# Chapter 2
## End-to-End Machine Learning Project


### General Steps to Build a Machine Learning Project
1. [Frame the problem and look at the big picture.](#1-frame-the-problem-and-look-at-the-big-picture)
2. [Get the data.](#2-get-the-data)
3. [Discover and visualize the data to gain insights.](#3-discover-and-visualize-the-data-to-gain-insights)
4. [Prepare the data to better expose the underlying data patterns to Machine Learning algorithms.](#4-prepare-the-data-to-better-expose-the-underlying-data-patterns-to-machine-learning-algorithms)
5. [Explore many different models and shortlist the best ones.](#5-explore-many-different-models-and-short-list-the-best-ones)
6. [Fine-tune your models and combine them into a great solution.](#6-fine-tune-your-models-and-combine-them-into-a-great-solution)
7. Present your solution.
8. Launch, monitor, and maintain your system.

### Where to Get Real Data
- [UC Irvine Machine Learning Repository](http://archive.ics.uci.edu/ml/)
- [Kaggle datasets](https://www.kaggle.com/datasets)
- [Amazon's AWS datasets](https://registry.opendata.aws/)
- [Data Portals](http://dataportals.org/)
- [OpenDataMonitor](opendatamonitor.eu)
- [Quandl](https://data.nasdaq.com/)
- [Wikipedia's List of Machine Learning datasets](https://en.wikipedia.org/wiki/List_of_datasets_for_machine-learning_research)
- [Quora.com](https://www.quora.com/Where-can-I-find-large-datasets-open-to-the-public)
- [The datasets subreddit](https://www.reddit.com/r/datasets)


### Machine Learning Housing Corporation Project
#### **1. Frame the problem and look at the big picture.**
Contents
- [1A Look at the Big Picture](#1a-look-at-the-big-picture)
- [1B Frame the Problem](#1b-frame-the-problem)
- [1C Select a Performance Measure](#1c-select-a-performance-measure)
- [1D Check the Assumptions](#1d-check-the-assumptions)

#### 1A Look at the Big Picture
Build a model of California housing prices that can predict the median housing price in any district (block group) given the following metrics,
- Population
- Median income
- Median housing price

#### 1B Frame the Problem
The model output is to be fed into another machine learning system with other signals whose output will be used to determine whether it is worth investing in a given area which directly affects company revenue.
##### def. signal - a piece of information fed to a Machine Learning system

#### Pipelines
##### def. pipeline - sequence of data processing components
Components of a pipeline run asynchronously. Each pulls data, processes it, and outputs it to another data store. After some time the next component performs its operations. Each component is self-contained, i.e. components interface using the data store. This makes the architecture simple and robust. This could make it more likely that a broken component will go unnoticed dropping the overall systems performance due to stale data.

#### Questions to Ask Before Framing the Problem
1. What is the business objective?
2. What does the current solution look life (if any)?

#### Questions to Ask While Designing the System to Frame the Problem
1. Should it learn supervised? Unsupervised? By reinforcement?
2. Is it a classification task? Regression? Something else?
3. Should it use a batch learning technique? Or online learning?

#### Framing the Problem for the Housing Corporation Project
1. Since the data is labeled (this is the training examples), such that each instance comes with expected output, it should use a supervised task.
2. Since a value will be predicted from the model, it should use a regression task, more specifically, since it is dealing with multiple features it will be a multiple regression task. It is also a univariate regression problem since only a single value will be made for a prediction (as opposed to a multivariate regression problem). 
3. Since there is no continuous flow of data, there is no need to adjust to changing data rapidly, and the data is small enough to fit in memory, so batch learning should be implemented (large amounts of data can be split across servers using MapReduce technique or online learning).


#### 1C Select a Performance Measure
Typical performance measures for regression problems is the *Root Mean Square Error* (RMSE) that gives an idea of how much error the system makes in its predictions with a higher weight for large errors.

$$
\begin{align}
    RMSE(\bf{X},\it{h}) = \sqrt{\frac{1}{\it{m}}\sum_{i=1}^{\it{m}}(\it{h}(\bf{x}^{i}) - \it{y}^{i})^{2}}
\end{align}
$$

Notation
- $\it{m}$: the number of instances in the dataset the RMSE will be measuring
- $\bf{x}^{i}$: a vector of all feature values of the $i^{th}$ instance in the dataset
- $\it{y}^{i}$: the label or desired output for the instance containing $\bf{x}^{i}$
- $\bf{X}$: a matrix containing all feature values (or all instances in the dataset) where each row is an instance, and the $i^{th}$ row is equal to the transpose of $\bf{x}^{i} = (\bf{x}^{i})^{T}$
- $\it{h}$: the hypothesis (or system's prediction function) that outputs a predicted value $\hat{\it{y}} = \it{h}(\bf{x}^{i})$ given an instance's feature vector $\bf{x}^{i}$
- $RMSE(\bf{X},\it{h})$: the cost function measured on the set of examples using $\it{h}$

Note: $\it{italic}$ font is used for scalar values and function names, $\bf{bold}$ font is used for vectors and matrices

Another performance measure that could be used if there are many outliers is the *Mean Absolute Error* (MAE).

$$
\begin{align}
    MAE(\bf{X},\it{h}) = \frac{1}{m}\sum_{i=1}^{\it{m}}|h(\bf{x}^i - \it{y}^i)|
\end{align}
$$

Both RMSE and MAE measure the distance between the prediction vector and the target value vector. The following are possible variations of distance measures (or norms),

- *Euclidean Norm* is the distance corresponding to the RMSE called the $l_1$ *norm*, denoted by $||\cdot||_2$ (or simply $||\cdot||$).
- *Manhattan Norm* is the distance between two points such that the lines between points can only be orthogonal corresponding to the sum of absolutes (MAE), denoted by $||\cdot||_1$.
- $l_k$ *norm*, is the distance of a vector $\bf{v}$ containing $\it{n}$ elements, defined as $||\bf{v}||_k = (|\it{v}_0|^k + |\it{v}_1|^k + ... + |\it{v}_n|^k)^{\frac{1}{k}}$, where $l_0$ gives the number of nonzero elements in the vector and $l_\infty$ gives the maximum absolute value in the vector.
- The higher the norm index value, *k*, the more focused the measure is on large values neglecting smaller ones, making RMSE more sensitive to outliers than MAE; but RMSE is preferred when outliers are exponentially rare (bell-shaped curve) which improves its performance.

#### 1D Check the Assumptions
Verify assumption to catch serious issues early on.

It is assumed that prices will be fed into the downstream Machine Learning system and used in this numeric form. However, if the prices are converted into categories (e.g. "cheap", "medium", "expensive") then getting prices perfectly is not important. This would change the problem framing from a regression task to a classification task.

In the Housing Corporation Project, assume actual prices are needed.

#### **2. Get the Data**
Contents
- [2A Create the Workspace](#2a-create-the-workspace)
- [2B Download and Load the Data](#2b-download-and-load-the-data)
- [2C Look at the Data Structure](#2c-look-at-the-data-structure)
- [2D Create a Test Set](#2d-create-a-test-set)

Data Repositories
- [Full Hands-On ML Jupyter Notebook](https://github.com/ageron/handson-ml2)
- [Housing Corporation Project Data](https://github.com/ageron/handson-ml2/tree/master/datasets/housing)

#### 2A Create the Workspace
1. [Install Python](https://www.python.org/)
2.  Create a workspace directory for the ML code and datasets
    ```console
    $ export ML_PATH="$HOME/hands-on-ml"
    $ mkdir -p $ML_PATH
    ```
3. Create a virtual environment and activate
    ```console
    $ python3 -m venv .
    $ source .venv/bin/activate
    ```
4. Create requirements.txt for Jupyter, Matplotlib, NumPy, pandas, SciPy, and Scikit-Learn
    ```txt
    jupyter>=1.0.0
    matplotlib>=3.7.1
    numpy>=1.24.2
    pandas>=1.5.3
    scipy>=1.10.1
    scikit-learn>=1.2.2
    ```
5. Install modules
    ```console
    $ pip3 install -r requirements.txt
    ```
6. Register venv to Jupyter
    ```console
    $ python3 -m ipykernel install --user --name=python3
    ```
7. Open Jupyter Notebook
    ```console
    $ jupyter notebook
    ```
8. Access notebook at http://localhost:8888/
9. Create a new Python notebook.

#### 2B Download and Load the Data
Data is typically available in relational databases or spread over multiple tables, documents, or files which require credentials and access authorizations.

The data for the Housing Corporation Project is stored in a single csv file. This process is automated by making a request from the [Hands-On ML GitHub Repository](https://github.com/ageron/handson-ml2/tree/master/datasets/housing) as a compressed `.tgz` file and extracts it into the project directory (see: `fetch_housing_data` in [notebook](https://github.com/hiyaryan/hands-on-ml/blob/main/src/ch_2/Housing.ipynb)).

Load the data into a pandas `DataFrame` (see: `load_housing_data` in [notebook](https://github.com/hiyaryan/hands-on-ml/blob/main/src/ch_2/Housing.ipynb)).

#### 2C Look at the Data Structure
`head()` 

Allows you to look at the top 5 rows of the DataFrame. Each column is an attribute. There are a total of 10 attributes in the housing dataset:

`longitude`, `latitude`, `housing_median_age`, `total_rooms`, `total_bedrooms`, `population`, `households`, `median_income`, `median_house_value`, `ocean_proximity`

`info()` 

Gives you a description of the data such as the total number of rows, the attributes type, and number of non-null values. 
- There are 20,640 instances. 207 of the instances are missing the `total_bed_rooms` feature.
- All attributes are numerical except the `ocean_proximity` field of type `object` which must be a text attribute since the data came from a CSV file.
  - From the `head` these rows are repetitive indicating they are likely categorical.
  - `value_counts()` shows how many categories exist and how many districts belong to each category.

`describe()`

Allows you to view a summary of the *numerical* attributes.

The summary contains the count, mean, std, min, max, and percentiles of each field.
- std (Standard Deviation) measures how dispersed the values are.
- 25%, 50%, 75% percentiles indicate the value under which a given percentage of the instances in a group of observations fall

Note: The standard deviation is denoted by $\sigma$ and is the square root of the variance which is the average of the square deviation from the mean. When a feature has a bell-shaped normal distribution (or Gaussian distribution) the "68-95-99.7" rule applies: 68% of the values fall within $1\sigma$ of the mean, 95% fall within $2\sigma$, and 99.7% within $3\sigma$.

`hist(...)`

Allows you to plot a histogram that can show how many instances along the vertical axis have a given value along the horizontal axis.

Housing data histogram analysis:

1.  `median_income` has been preprocessed. It is scaled and capped at 15.0001 for higher median incomes and at 0.4999 for lower median incomes where each number represents roughly tens of thousands of dollars.
2.  `housing_median_age` and `median_house_value` are also capped. Since `median_house_value` is the target attribute, the ML algorithm may learn that prices never go beyond that limit. If precise predictions are required then proper labels for the districts whose labels were capped should be collected, or remove the districts that are capped from the training set.
3.  The attributes are scaled differently. This means the ML algorithm will likely not perform well without applying a *feature scaling* transformation.
4.  Many of the histograms are *tail-heavy*, meaning they extend farther to the right of the median than to the left potentially making it harder for some ML algorithms to detect patterns. These attributes can possibly be transformed to have more *bell-shaped* distributions.

Note: If importing `matplotlib.pyplot` in a Jupyter Notebook, the magic command, `%matplotlib inline`, may be required in order to render the plot inline in the notebook using Jupyter Notebook's own graphical backend, otherwise, a user-specified graphical backend must be selected.

#### 2D Create a Test Set
To prevent the *data snooping bias*, which can introduce an overly optimistic estimate of the generalization error, it is best practice to set aside part of the data before deciding on a Machine Learning model. This bias occurs when the brain selects a Machine Learning model after having stumbled upon some interesting pattern in the data. The issue is that the brain's pattern detection system is prone to overfitting, which can hurt the performance of the system.

A test set should be a random selection of instances set aside that has a size of about 20%, or smaller depending on the length, of the dataset.

```py
def split_train_test(data, test_ratio):
    '''
    Splits the dataset into a test set and train set.
    '''
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]
```

The test set should always contain the same instances, otherwise, a new test set will be generated every time it runs and the entire dataset will be seen eventually. The `split_train_test` function assumes it will run only once and that the dataset will not be updated. To solve for this, the test set can be saved separately from the dataset and loaded into every subsequent run or, the random number generator's seed can be set so that the same shuffled indices are always selected. 

In addition to the test set containing the same instances, the dataset should be able to be updated with new fetched data. This can be done by computing a hash of each instance's identifier and put the instance in the test set if the hash is lower than 20% of the maximum hash value ensuring consistency across multiple runs even if the dataset is refreshed. This means the new test set will contain 20% of the new instances and none of the instances in the previous test set.

```py
def test_set_check(identifier, test_ratio):
    '''

    '''
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32


def split_train_test_by_id(data, test_ratio, id_column):
    '''
    '''
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc(~in_test_set), data.loc[in_test_set]
```

These functions require an identifier. Since the housing data does not have one, the row index can be used as long as new data is only appended to the end, otherwise, more stable features can be combined to create an ID e.g. `longitude` and `latitude`.

```py
housing_with_id = housing.reset_index()
housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")
```

Scikit-Learn also provides some functions to split datasets into subsets. A similar simple function to `split_train_test` is `train_test_split` that can take a `random_state` parameter that allows a random generator seed which can ensure only the same test set is extracted from the dataset.

```py
# use of Scikit-Learn to split the dataset into a train and test set
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
```

A *sampling bias* can occur if the dataset is not large enough and the test set is selected through random sampling methods. This is because if the right number of instances are not sampled from each stratum (def. strata - a homogenous subgroup of a population), then the test set may not be representative of the overall population. *Stratified sampling* can be used to ensure representative samples are selected.

In the housing dataset, `median_income` is found to be an important attribute to to predict `median_house_value` so the test set should be representative of a stratum of `median_income`. This requires stratified sampling which can be performed through analyzing the `median_income` histogram and determine how many of each instance from each bucket should be included in the test set. The size of the bucket determines the size of each strata in the test set.

#### **3. Discover and visualize the data to gain insights.**
Contents
- [3A Visualizing Geographical Data](#3a-visualizing-geographical-data)
- [3B Looking for Correlations](#3b-looking-for-correlations)
- [3C Experimenting with Attribute Combinations](#3c-experimenting-with-attribute-combinations)

Put the *test set* aside and explore the *training set*. If the training set is large another *exploration set* be sampled.

#### 3A Visualizing Geographical Data
Geographical data can be visualized with scatterplots. 

```
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
```

`alpha` setting alpha makes it easier to visualize high density points on the plot.

Scatter Plot Options
- `s` radius of a circle 
- `c` color gradient 
- `cmap` color map (blue are low values, red are high values)

```
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, s=housing["population"]/100, label="population", figsize=(10, 7), c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
```

This housing scatter plot shows that the prices are related to the location and population density.

A clustering algorithm can be useful to detect the main cluster and for adding new features that measure the proximity to the cluster centers.

#### 3B Looking for Correlations
Since the dataset is not large, the *standard correlation coefficient*, or *Pearson's r*, can be computed using the `corr()` function on the DataFrame.

Correlation coefficients range from [-1, 1], where 1 is a strong positive correlation, 0 is no correlation, and -1 is a strong negative correlation. 

Pandas also has a `scatter_matrix` function that plots every numerical attribute against every other numerical attribute. This can be useful to spot patterns and to detect attributes that have a strong correlation.

#### 3C Experimenting with Attribute Combinations
Before preparing data for Machine Learning algorithms, it is useful to try out various attribute combinations to see if they can improve the accuracy of the model. 

Some combinations worth experimenting in the housing dataset with are rooms per household, bedrooms per rooms, and population per household.

By adding these combinations to the dataset, the correlation matrix can be computed again to see if the new attributes have a higher correlation with the `median_house_value`. 

Through experimentation, it was found that the `rooms_per_household` attribute is more correlated with the `median_house_value` than the `total_rooms` attribute.

This step is iterative and can be returned to after the model is trained to see if the model performs better with the new attributes that may be discovered with more experimentation.

#### **4. Prepare the data to better expose the underlying data patterns to Machine Learning algorithms.**
Contents
- [4A Data Cleaning](#4a-data-cleaning)
- [4B Handling Text and Categorical Attributes](#4b-handling-text-and-categorical-attributes)
- [4C Custom Transformers](#4c-custom-transformers)
- [4D Feature Scaling](#4d-feature-scaling)
- [4E Transformation Pipelines](#4e-transformation-pipelines)

Prepare the data for Machine Learning algorithms by cleaning the data. This can be done by writing functions that fill in missing values, remove outliers, and convert non-numerical values to numerical values.

#### 4A Data Cleaning
Missing features can break Machine Learning algorithms so it is important to handle them. To handle the missing values found in `total_bedrooms` there are three options:
1. Get rid of the corresponding districts
```py
housing.dropna(subset=["total_bedrooms"])
```
2. Get rid of the whole attribute
```py
housing.drop("total_bedrooms", axis=1)
```
3. Set the values to some value (zero, the mean, the median, etc.)
```py
median = housing["total_bedrooms"].median()
housing["total_bedrooms"].fillna(median, inplace=True)
```

Scikit-Learn provides a `SimpleImputer` class to handle missing values. 

#### 4B Handling Text and Categorical Attributes
Most Machine Learning algorithms prefer to work with numbers so text and categorical attributes should be converted to numbers.

The `ocean_proximity` attribute is a categorical attribute with text values. This can be converted to numerical values by using the `OrdinalEncoder` class from Scikit-Learn.

To prevent the algorithm from assuming that two nearby values are more similar than two distant values, a *one-hot encoding* can be used. This can be done with the `OneHotEncoder` class from Scikit-Learn.

##### def. one-hot encoding - creating a new binary attribute for each category and assigning a 1 (hot) or 0 (cold) to the attribute depending on the category of the instance

##### def. dummy attribute - new binary attribute created by one-hot encoding

For a large number of possible categories, one-hot encoding can result in a large number of input features which can slow down training and degrade performance. Some solutions to this problem are:
- Replacing the categorical attribute with a useful numerical attribute.
- Replacing each category with a learnable, low-dimensional vector called an *embedding* that is learned during training (this is called *representation learning*).

#### 4C Custom Transformers
Scikit-Learn relies on duck typing (not inheritance) so in order to use a custom transformer that works seamlessly in a Scikit-Learn pipeline a class can be created that implements three methods: `fit()`, `transform()`, and `fit_transform()`. These methods perform tasks such as custom cleanup operations or combining specific attributes.

Two Scikit-Learn classes can be used to create a custom transformer: `BaseEstimator` and `TransformerMixin`. The `BaseEstimator` class gives the `get_params()` and `set_params()` methods, useful for hyperparameter tuning, and the `TransformerMixin` class adds the `fit_transform()` method.

The two methods that need to be implemented are `fit()` and `transform()`. The `fit()` method is used to learn the parameters of the transformation, and the `transform()` method is used to apply the transformation to the data.

A hyperparameter can be added to any data preparation step that has some uncertainty in how it should be prepared. For example, the `add_bedrooms_per_room` hyperparameter can be added to the `CombinedAttributesAdder` class to determine whether or not to add the new attribute.

#### 4D Feature Scaling
Feature scaling is the process of transforming numerical attributes so that they have the same scale. This is important for Machine Learning algorithms that do not perform well when the input numerical attributes have very different scales.

For example, the `total_rooms` attribute ranges from about 6 to 39,320, while the `median_income` attribute ranges from 0 to 15. The `total_rooms` attribute has a much larger scale than the `median_income` attribute.

Two common ways to get all attributes to have the same scale are *min-max scaling* and *standardization*.

##### def. min-max scaling - also called normalization, this is the process of subtracting the min value and dividing by the max minus the min shifting the range of values to be between 0 and 1 (can be done with Scikit-Learn's `MinMaxScaler` class)

##### def. standardization - this is the process of subtracting the mean value and dividing by the variance so that the resulting distribution has unit variance which does not bound values to a specific range making it less affected by outliers (can be done with Scikit-Learn's `StandardScaler` class)

Ensure that all transformations are applied to the training set and not the full dataset. This is to prevent data leakage which can occur when the test set is used to make decisions about the data preparation process (such as deciding which transformations to apply).

#### 4E Transformation Pipelines
Data transformations must be executed in the right order. Scikit-Learn provides the `Pipeline` class to help with this. 

The `Pipeline` constructor takes a list of name/estimator pairs defining a sequence of steps. All but the last estimator must be transformers (i.e. they must have a `fit_transform()` method). The names can be anything you like.

`fit_transform()` is called on all transformers in the pipeline, in the order they were specified, and the result is passed to the next transformer's `fit_transform()` method until the final estimator is reached, at which point the pipeline stops and the final estimator's `fit()` method is called.

`ColumnTransformer` can be used to apply different transformations to different columns. This is useful for the housing dataset because the numerical attributes need to be scaled while the categorical attributes need to be converted to one-hot vectors.

If there is a mixture of sparse and dense features, the `ColumnTransformer` estimates the density of the final matrix (i.e. the ratio of nonzero cells), and it returns a sparse matrix if the density is lower than a given threshold (by default, `sparse_threshold=0.3`). 

#### **5. Explore many different models and short-list the best ones.**
Contents
- [5A Training and Evaluating on the Training Set](#5a-training-and-evaluating-on-the-training-set)
- [5B Better Evaluation Using Cross-Validation](#5b-better-evaluation-using-cross-validation)

#### 5A Training and Evaluating on the Training Set
First, train a Linear Regression model. This is a good model to start with because it is fast to train and it is easy to understand and will serve as a baseline for the more complex models.

Use RMSE as the performance measure for the Linear Regression model. Recall this is a common performance measure for regression problems. The RMSE of a model is the square root of the mean of the squared errors.The `mean_squared_error()` function from Scikit-Learn can be used to calculate the MSE. The `np.sqrt()` function can be used to calculate the RMSE.

For the housing dataset, a Linear Regression model underfits the data. This is because the features do not provide enough information to make good predictions, or the model is not powerful enough.

Next, train a Decision Tree model. This is a powerful model, capable of finding complex nonlinear relationships in the data. The Decision Tree model is capable of finding patterns in the data that Linear Regression model is unable to find.

The Decision Tree model is overfitting the data. This is because the model is too complex and has memorized the noise in the training data.

#### 5B Better Evaluation Using Cross-Validation
The Decision Tree model can be evaluated using the `train_test_split()` function to split the training set into a smaller training set and a validation set, then train the model against the smaller training set and evaluate it against the validation set. This process can be repeated several times using different validation sets to get an idea of how the model will perform on new data.

Scikit-Learn has a K-fold cross-validation feature that can train and evaluate the Decision Tree model 10 times (by default) using different combinations of the training and validation sets. The result is an array containing the 10 evaluation scores.

Cross-validation allows you to get both a measure of the model's performance and a measure of how precise this performance measure is. The standard deviation of the scores is a measure of how precise the estimate is.

Compared to the Decision Tree model, the Linear Regression model performs better. This is because the Decision Tree model is overfitting the data worse than the Linear Regression model underfitting it. 

Another model to try is the Random Forest model that is a type of ensemble learning. This model works by training many Decision Trees on random subsets of the features, then averaging out their predictions.

##### def. Ensemble Learning - the process of combining the predictions of several models to get better predictions than with any individual model. The Random Forest model is an example of an ensemble learning method called *Random Forests*.

The Random Forest model performs better than the Linear Regression model. However, it is still overfitting the training data.

Another model to try is the Support Vector Machine (SVM) model. This model is particularly good at performing linear regression. The SVM model is also good at handling complex nonlinear datasets.

Finally try a neural network model. This model is capable of learning complex nonlinear relationships in the data. However, it is very slow to train and it requires a lot of data to perform well.

Use Python's `pickle` module to save the trained models. This is useful for when you want to come back to a project later and pick up where you left off.

#### **6. Fine-tune your models and combine them into a great solution.**
Contents
- [6A Grid Search](#6a-grid-search)
- [6B Randomized Search](#6b-randomized-search)
- [6C Ensemble Methods](#6c-ensemble-methods)
- [6D Analyze the Best Models and Their Errors](#6d-analyze-the-best-models-and-their-errors)
- [6E Evaluate Your System on the Test Set](#6e-evaluate-your-system-on-the-test-set)

Hyperparameters are parameters that are not directly learnt within estimators. They can be set manually until the model performs well or they can be tuned automatically using techniques such as grid search or randomized search.

#### 6A Grid Search
SciKit-Learn's `GridSearchCV` class can be used to search for the best hyperparameter values. It is given a dictionary of hyperparameters and it tries out all the possible combinations of values, using cross-validation to evaluate each model.

When the hyperparameter value is unclear try consecutive powers of 10. For example, if you are not sure whether the `n_estimators` hyperparameter should be set to 10, 100, or 1000, try setting it to 10, 100, and 1000 to see which works best.

View the best hyperparameter values found by the grid search using the `best_params_` attribute and the best estimator using the `best_estimator_` attribute. View the evaluation scores for each hyperparameter combination tested during the grid search using the `cv_results_` attribute and search for the lowest RMSE score to find the best hyperparameter combination.

Grid search is good for exploring relatively few combinations, but when the hyperparameter search space is large, it is often preferable to use `RandomizedSearchCV` instead.

#### 6B Randomized Search
Randomize search can be used when the hyperparameter search space is large. It evaluates a given number of random combinations by selecting a random value for each hyperparameter at every iteration. Benefits of randomized search include:
- Exploring a larger specified hyperparameter space.
- Giving more control over the computing budget.

#### 6C Ensemble Methods
Ensemble methods fine-tune models by combining them into an ensemble. The ensemble's predictions are often better than the best individual model's predictions.

#### 6D Analyze the Best Models and Their Errors
Insights can be gained by analyzing the best models and their errors. `RandomForestRegressor` can be used to get the relative importance of each attribute for making accurate predictions.

When analyzing the best models and their errors, it's useful to drop the least useful features. Additionally, look at the specific errors the system makes and try to understand why it makes them.

#### 6E Evaluate Your System on the Test Set
To evaluate the final model on the test set, run the `test_set` through the `full_pipeline` to transform the data (call `transform()`, not `fit_transform()`), then evaluate the final model on the transformed test set.

Ensure the test set is transformed using `transform()` and not `fit_transform()` to avoid data leakage which will fit the model to the test set.

To get an idea of how precise the model's generalization error is, compute the 95% confidence interval for the generalization error using `scipy.stats.t.interval()`.

After evaluating the final model, resist any more fine-tuning which likely won't improve the model's performance on new data. Instead, collect more data and try different models or prepare to launch, monitor, and maintain the system.