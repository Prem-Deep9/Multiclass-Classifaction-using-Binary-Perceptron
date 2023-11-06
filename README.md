# Project Objectives 
- Implementing Binary Perception from scratch using Python.
- Achieving Multiclass Classification on given data, Using One-vs-One and One-vs-Rest approaches.
- Implementing Regularisation in the One-vs-Rest approach, observing the accuracies for multiple Regularisation coefficient values.

# Binary Perceptron
The Perceptron algorithm is one of the earliest supervised machine learning techniques used to classify data points into two groups. This algorithm was invented by Frank Rosenblatt in 1958 and was used to identify different types of shapes. A perceptron is a model of a single neuron.\
The inputs are multiplied with respective weights and summed up giving us net input. Bias is added to the net input to calculate the activation score. The activation score is then compared with a threshold from the activation function we choose. If greater than the threshold i.e. zero in our case, when the activation score is greater than 0 we say the output is 1 or positive and if it is less than zero we output is -1 or negative.
When a correct classification happens the weights remain unchanged, when a wrong classification occurs the weights are updated by the formula wi = wi+ y ⋅ xi where y is the desired output and bias is updated with the formula b = b + y. The model runs for all inputs updating weights and biases at every stage. When all inputs and iterations are completed we get a trained set of weights and bias. Which can be used to test data for accuracy.\
Input : XT = (x1, x2, …, xd), weights = WT = (w1,w2, …,wd)\
Activation score a = $\\sum\limits_{i=1}^{d} w_i x_i = \overline{W^T X}$\
<img src="https://github.com/Prem-Deep9/Perceptron/blob/main/Binary%20Perceptron.png" height="400" width="600">\
In our code, Initially, the weights are initialized as zero for the sake of simplicity, for different problems they can be initialized differently. The Bias I initialized next, can be initialized as zero as well. The Algorithm processes objects from the training data set one by one (as opposed to batch learning that requires access to the entire data set, e.g. k-NN).\
Error driven: the parameters are updated only when a test object is classified wrongly using the current parameters (weights and bias).

### Pseudo Code:
PerceptronTrain (Training data: D, MaxIter)
1. w<sub>i</sub> = 0 for all i = 1,...,d
2. b = 0
3. for iter = 1 ... MaxIter do
4. &nbsp;&nbsp;&nbsp;&nbsp;for all ($\overline{W^T X})\in D$ do
5. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;a = $\overline{W^T X}$ + b
6. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;if y . a <= 0 then
7. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;w<sub>i</sub> = w<sub>i</sub> + y . x<sub>i</sub>, for all i = 1, ... , d
8. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;b = b + y
9. return b, w<sub>1</sub>, w<sub>2</sub>, ... , w<sub>d</sub>

PerceptronTest (b, w<sub>1</sub>, w<sub>2</sub>, ... , w<sub>d</sub>)
1. a = $\overline{W^T X}$ + b
2. return sign(a)

# One-vs-One Approach
In this approach, a binary classifier is trained for every pair of classes. The idea is to classify the data into one of the two classes in each binary classifier. When you want to make a prediction, you apply all classifiers to the data point and use a voting mechanism (e.g., majority voting) to determine the final class. Here's how it works:
1. For each pair of classes i and j use training objects from classes i and j to train
algorithm A to distinguish between objects in classes i and j. Denote the obtained
classifier A<sub>ij</sub>
2. For K classes, we train k(k-1)/2 prediction models.
3. Applying the prediction model A<sub>ij</sub> to an incoming object $\bar{X}$ is interpreted as voting. A<sub>ij</sub> votes either +1 for $\bar{X}$ to be in class i, or A<sub>ij</sub> votes +1 for $\bar{X}$ to be in class j.
4. For an incoming object $\bar{X}$, apply all prediction models one by one.
5. The class label with the most votes is declared as the winner.
### Drawbacks
There might be ambiguity if some classes got the same number of votes (if the binary classifier can produce a confidence score, it can be used to break ties)

# One-vs-Many Approach

In this approach, we assume that the binary classification algorithm A can output a numeric score representing its “confidence” that an object belongs to a particular class. We train K binary classifiers, where K is the number of classes. Each binary classifier is responsible for distinguishing one class from the rest of the classes. When making a prediction, you apply all K classifiers to the data point, and the class with the highest confidence score is the predicted class. Here's how it works:
1. For each class i, train the binary classifier A with the objects of the class i treated as positive samples and all other objects as negative samples. Denote the obtained classifier A<sub>i</sub>.
2. This results in K prediction models.
3. For an incoming object $\bar{X}$, apply all prediction models A<sub>1</sub>,A<sub>2</sub>,...,A<sub>K</sub>. 
4. Output for object $\bar{X}$ the class label y corresponding to the model with the highest score.

The choice of the numeric score depends on the classifier at hand.
1. For Perceptron: the activation score a = $\overline{W^T X}$ + b
2. For Logistic regression: σ(a), where a = $\overline{W^T X}$ + b

### Drawbacks
- The scale of the confidence scores may differ between the binary classifiers
- The binary classifiers are trained on unbalanced datasets: usually, the set of negative objects will be much larger than the set of positive objects

# Regularisation

- Regularisation is a process of reducing overfitting in a model by constraining it (reducing the complexity/no. of parameters)
- For classifiers that use a weight vector, regularisation can be done by minimizing the norm (length) of the weight vector.
- Several popular regularisation methods exist
- L2 regularisation (ridge regression or Tikhonov regularisation)
- L1 regularisation (Lasso regression)
- L1+L2 regularisation (mixed or Elastic Net regularisation)

# Guide for running the code
1. Install the modules Numpy and Pandas.
2. Replace the path mentioned in the code near pd.read_csv wherever it appears with the path to test and train data files respectively on your computer.
3. For One-vs-One approach class instance of Perceptron needs following input variables in order: datasets(pass 'train_1vs2' to discriminate between class-1 and class-2, pass 'train_2vs3' to discriminate between class-2 and class-3, pass 'train_1vs3' to discriminate between class-1 and class-3), initial weights (list of length 4), bias(integer) and iterations(integer) as input to the class object. After initializing the class Object, call the methods train with respective preprocessed data and test on each data point of the test data.
5. For One-vs-Rest approach class instance of Perceptron needs the following input variables in order: datasets(pass 'train_1vsRest' to discriminate between class-1 and rest, pass 'train_2vsRest' to discriminate between class-2 and rest, pass 'train_3vsRest' to discriminate between class-3 and rest), initial weights (list of length 4), bias(integer) and iterations (integer) as input to the class object. After initializing the class Object, call the methods train with respective preprocessed data and multi_test on each data point of the test data.
6. One-vs-rest with Regularisation uses Perceptron_Regularisation class. The class object is similar to Perceptron, the Only difference being adding a regularisation coefficient parameter to initialize the class object. The approach is similar to One-vs-Rest.
7. Class predictions and overall accuracies are printed onto the terminal.
8. Sample datasets and class instances are provided, you can run the code directly to see results with default inputs.

# Results for test data with 20 epochs and initial weights & bias =0 and Regularisation_Coefficient = [.01, 0.1, 1.0, 10.0, 100.0]
Accuracy of one vs one approach to classify the data is 100.0\
Accuracy of one vs Rest approach to classify the data is 73.33333333333333\
Accuracy of one vs Rest with Regularisation Coefficient 0.01 to classify the data is 70.0\
Accuracy of one vs Rest with Regularisation Coefficient 0.1 to classify the data is 66.66666666666666\
Accuracy of one vs Rest with Regularisation Coefficient 1.0 to classify the data is 30.0\
Accuracy of one vs Rest with Regularisation Coefficient 10.0 to classify the data is 33.33333333333333\
Accuracy of one vs Rest with Regularisation Coefficient 100.0 to classify the data is 33.33333333333333

# Inference
- small changes in the training data significantly impact a model's accuracy, it is a clear indication of high variance or overfitting. High variance means that the model is overly sensitive to the specific data points it was trained on, and it fails to generalize well to new, unseen data.
- A binary perceptron, like any other machine learning model, can exhibit high variance when trained on a small amount of data. Variance, in the context of machine learning, refers to the model's sensitivity to the specific training data it is provided.
- Ways to improve: Collect More Data, Cross-Validation, Hyperparameter Tuning: epochs, etc., initial different initial weights and bias, Regularization, usage of complex models.
