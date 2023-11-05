Implementing Binary Perception from scratch using Python.

# Binary Perceptron
The Perceptron algorithm is one of the earliest supervised machine learning technique used to classify data points into two groups. This algorithm was invented by Frank Rosenblatt in 1958 and was used to identify different types of shapes. Perceptron is a model of a single neuron.\
The inputs are multiplied with respective weights and summed up giving us net input. Bias is added to the net input to calculate activation score. The activation score is the compared with a threshold from the activation function we choose. If greater than threshold or zero in our case, when activation score is greater than 0 we say output is 1 or positive and if it is less than zero we output is -1 or negative.
When correct classification happens the weights remain unchanged, when a wrong classification occurs the weights are updated by the formula wi = wi+ y ⋅ xi where y being the desired output and bias is updated with the formula b = b + y. The model runs for all inputs updating weights and biases at every stage. When all inputs and iterations are completed we get a trained set of weights and bias. Which can be used to test data for accuracy.\
Input : XT = (x1, x2, …, xd), weights = WT = (w1,w2, …,wd)\
Activation score a = $\\sum\limits_{i=1}^{d} w_i x_i = \overline{W^T X}$\
<img src="https://github.com/Prem-Deep9/Perceptron/blob/main/Binary%20Perceptron.png" height="400" width="600">


# One-vs-One Approach
In this approach, a binary classifier is trained for every pair of classes. The idea is to classify the data into one of the two classes in each binary classifier. When you want to make a prediction, you apply all classifiers to the data point and use a voting mechanism (e.g., majority voting) to determine the final class. Here's how it works:
1. For each pair of classes i and j use training objects from classes i and j to train
algorithm A to distinguish between objects in classes i and j. Denote the obtained
classifier A<sub>ij</sub>
2. For K classes, we train k(k-1)/2 prediction models.
3. Applying the prediction model A<sub>ij</sub> to an incoming object $\bar{X}$ is interpreted as voting. A<sub>ij</sub> votes either +1 for $\bar{X}$ to be in class i, or A<sub>ij</sub> votes +1 for $\bar{X}$ to be in class j.
4. For an incoming object $\bar{X}$, apply all prediction models one by one.
5. The class label with the most votes is declared as the winner.

# One-vs-Many Approach

In this approach, we assume that the binary classification algorithm A can output a numeric score representing its “confidence” that an object belongs to a particular class. We train K binary classifiers, where K is the number of classes. Each binary classifier is responsible for distinguishing one class from the rest of the classes. When making a prediction, you apply all K classifiers to the data point, and the class with the highest confidence score is the predicted class. Here's how it works:
1. For each class i, train the binary classifier A with the objects of the class i treated as positive samples and all other objects as negative samples. Denote the obtained classifier A<sub>i</sub>.
2. This results in K prediction models.
3. For an incoming object $\bar{X}$, apply all prediction models A<sub>1</sub>,A<sub>2</sub>,...,A<sub>K</sub>. 
4. Output for object $\bar{X}$ the class label y corresponding to the model with the highest score.

# Regularisation

- Regularisation is a process of reducing overfitting in a model by constraining it (reducing the complexity/no. of parameters)
- For classifiers that use a weight vector, regularisation can be done by minimizing the norm (length) of the weight vector.
- Several popular regularisation methods exist
- L2 regularisation (ridge regression or Tikhonov regularisation)
- L1 regularisation (Lasso regression)
- L1+L2 regularisation (mixed or Elastic Net regularisation)
