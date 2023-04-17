# Chapter 1
## The Machine Learning Landscape

### Exercises
1. How would you define Machine Learning?

    Machine learning is the science (and art) of programming computers so they can learn from data. 

1. Can you name four types of problems where it shines?

    Analyzing images of products on a production line to automatically classify them, detecting tumors in brain scans, summarizing long documents automatically, creating a chatbot of personal assistant.

3. What is a labeled training set?

    A training set used in supervised learning that includes the desired solutions called labels.

4. What are the two most common supervised tasks?

    Classification, trains an algorithm on labeled data allowing it to classify new similar data. And regression, trains an algorithm on labeled data and features, called predictors, allowing it to predict some target numerical value.

5. Can you name four common unsupervised tasks?
    
    Dimensionality reduction, simplifying data without losing too much information. Anomaly detection, detecting unusual data in the dataset. Novelty detection, detecting new instances of data that looks different from the rest of the dataset. Association rule learning, discovering interesting relationships between attributes.

6. What type of Machine Learning algorithm would you use to allow a robot to walk in various unknown terrains?

    Reinforcement learning, providing the robot, or agent, the ability to observe its environment and get rewards based on its actions of which it uses to create its own strategy, or policy, to maximize the reward.

7. What type of algorithm would you use to segment your customers into multiple groups?

    Unsupervised learning, using a hierarchical clustering algorithm that subdivides the customers into multiple groups with similar preferences.

8. Would you frame the problem of spam detection as a supervised learning problem or an unsupervised learning problem?

    Both. Supervised learning makes it so that there is some assurance the algorithm will not label something important as spam. To remove the necessity to label email upfront, an unsupervised learning task that uses some form of clustering and anomaly detection can sort emails. This reduces the labeling work on the user who may occasionally pull regular email from spam inbox.

9.  What is an online learning system?

    Training a system incrementally by feeding it data instances sequentially, either individually through a continuous flow or in mini-batches, allowing the system to learn on the fly.

10. What is out-of-core learning?

    Training a system on a dataset that is larger than the amount of data it can fit in its own memory by loading part of the data, running the training steps, and repeating until all of the data has been processed. 

11. What type of learning algorithm relies on a similarity measure to make predictions?

    Instance-based learning. A system learns examples by heart, then generalizes new cases by comparing them to the learned examples.

12. What is the difference between a model parameter and a learning algorithm’s hyperparameter?

    A model parameter is a learned value that best fits a function, such as a linear function, over a set of data that can be used to make predictions on new data. A hyperparameter is preset value that is unchanged through learning, it is a constant model parameter, constraining the model to less degrees of freedom which reduces the complexity to find multiple model parameters and the risk of overfitting, called regularization.

13. What do model-based learning algorithms search for? What is the most common strategy they use to succeed? How do they make predictions?

    Model-based learning algorithms search for a way to make generalizations from a set of examples using a prediction function used to make predictions. The most common strategy used is to minimize the cost function that measures how poorly the system is making these predictions by finding the distance from a training example and the models prediction. Predictions are made by inputting new feature data into the prediction function that contains the parameter values found by the learning algorithm.

14. Can you name four of the main challenges in Machine Learning?

    Having too little training data, nonrepresentative training data, poor-quality data, and irrelevant features. 

15. If your model performs great on the training data but generalizes poorly to new instances, what is happening? Can you name three possible solutions?

    This is the challenge of overfitting the training data where the model overgeneralizes and detects patterns not only in the data but noisiness in the data itself. Some solutions are choosing a prediction function with fewer parameters, gathering more training data, or reducing the noise in the training data.

16. What is a test set, and why would you want to use it?

    The test set is used to test the predictions of the model. This is useful to evaluate the error of the model which indicates how well it will perform with new instances of similar data.

17. What is the purpose of a validation set?

    The purpose of a validation set, or holding part of the training set, is to avoid measuring generalization between multiple models using the test set. When the model is chosen, it will have been adapted using the test set, a validation set will ensure the chosen model performs well on new data.

18. What is the train-dev set, when do you need it, and how do you use it?

    The train-dev set can be a mixture of the training and validation sets that mostly includes data that is not perfectly representative of the data to be used in production. It helps to indirectly determine if poor performance is due to data mismatch. The train-dev set should be used after the model is trained on the training set. If the model performs poorly on the train-dev set then it likely overfit the training set which indicates that the model should be simplified, regularized, trained on more training data, or the data should be cleaned up. Then it will be obvious that if the model performs poorly on the validation set, the problem is from data mismatch.

19. What can go wrong if you tune hyperparameters using the test set?

    If hyperparameters are tuned using the test set, then the model will be adapted to that test set and will likely not perform well in production when it analyzes new data. This is where the validation set, a subset of the training set, can be held out to ensure that the hyperparameters chosen for the model does not only fit well because it was trained on the training set. 