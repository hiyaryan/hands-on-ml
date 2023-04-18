# Chapter 2
## End-to-End Machine Learning Project


### General Steps to Build a Machine Learning Project
1. Frame the problem and look at the big picture.
2. Get the data.
3. Discover and visualize the data to gain insights.
4. Prepare the data to better expose the underlying data patterns to Machine Learning algorithms.
5. Explore many different models and shortlist the best ones.
6. Fine-tune your models and combine them into a great solution.
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