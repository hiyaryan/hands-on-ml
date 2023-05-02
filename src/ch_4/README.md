# Chapter 4
# Training Models

Find notebook for this chapter [here](https://github.com/hiyaryan/hands-on-ml/blob/main/src/ch_4/training_models.ipynb).

Contents
1. [Linear Regression](#linear-regression)
2. [Gradient Descent](#gradient-descent)
3. [Polynomial Regression](#polynomial-regression)
4. [Learning Curves](#learning-curves)
5. [Regularized Linear Models](#regularized-linear-models)
6. [Logistic Regression](#logistic-regression)

---
### Packages

---
---
### Linear Regression
Contents
- [The Normal Equation](#the-normal-equation)
- [Computational Complexity](#computational-complexity)

Linear models make predictions by computing a weighted sum of input features, plus a constant called the bias term (or intercept term).

*Equation 4-1. Linear Regression model prediction*
$$\hat{y} = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \dots + \theta_n x_n = \theta^T \mathbf{x}$$

* $\hat{y}$ is the predicted value.
* $n$ is the number of features.
* $x_i$ is the $i^{th}$ feature value.
* $\theta_j$ is the $j^{th}$ model parameter

*Equation 4-2. Linear Regression model prediction (vectorized form)*
$$\hat{y} = h_\theta(\mathbf{x}) = \theta \cdot \mathbf{x}$$

* $\theta$ is the model's parameter vector
* $\mathbf{x}$ is the instance's feature vector
* $\theta \cdot \mathbf{x}$ is the dot product of the vectors
* $h_\theta$ is the hypothesis function

To train a linear regression model means to set its parameters to that the model best fits the training set. These chosen values can be evaluated using the Root Mean Square Error (RMSE) cost function to measure the performance of the model. 

The goal is to minimize the RMSE by finding the optimal values of $\theta$. Minimizing the Mean Square Error (MSE) also leads to the same result, and is simpler, since it does not involve square roots.

*Equation 4-3. MSE cost function for a Linear Regression model*
$$MSE(\mathbf{X}, h_\theta) = MSE(\theta) = \frac{1}{m} \sum_{i=1}^{m} (\theta^T \mathbf{x}^{(i)} - y^{(i)})^2$$

#### The Normal Equation
A *close-form solution*, or mathematical equation, used to minimize the cost function, i.e. find the value of $\theta$, is called the *Normal Equation*.

*Equation 4-4. Normal Equation*
$$\hat{\theta} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$

* $\hat{\theta}$ is the value of $\theta$ that minimizes the cost function.
* $\mathbf{X}$ is the vector of feature values containing $x^{(1)}$ to $x^{(m)}$.
* $\mathbf{y}$ is the vector of target values containing $y^{(1)}$ to $y^{(m)}$.


#### Computational Complexity
The computational complexity of the normal equation is $O(n^{2.4})$ to $O(n^3)$, depending on the implementation. This is due to the matrix inversion operation $\mathbf{X}^T \mathbf{X}$.

The SVD approach used by Scikit-Learn's `LinearRegression` class is about $O(n^2)$. If the number of features is doubled, the computation time is multiplied by roughly 4.

For very large number of features, the normal equation and SVD are very slow. However, because they are linear with respect to the number of training set instances, they handle large training sets efficiently, provided they can fit in memory.

Once linear regression models are trained, predictions are very fast since the computational complexity is linear with respect to both the number of instances to be predicted and the number of features. Making predictions on twice as many instances (or twice as many features) will about twice the time.