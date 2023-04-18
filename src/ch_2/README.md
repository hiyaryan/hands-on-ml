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
#### Look at the Big Picture
Build a model of California housing prices that can predict the median housing price in any district (block group) given the following metrics,
- Population
- Median income
- Median housing price

#### Frame the Problem
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