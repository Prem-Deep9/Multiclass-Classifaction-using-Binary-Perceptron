import numpy as np
import pandas as pd

class Perceptron:
     
    def __init__(self,input,weights,bias,n_iterations):
        self.n_iterations = n_iterations
        self.W = weights
        self.X = input
        self.bias = bias
    
    # new_weights updates weights and bias
    def new_weights(self,predicted_result,Xi):
        for i in range(len(self.W)):
            self.W[i] = self.W[i] + (Xi[i] *  predicted_result)
        self.bias = self.bias + predicted_result

    # train weights         
    def train(self):
        for iter in range(self.n_iterations):
            self.X = self.X.sample(frac=1).reset_index(drop = True)
            for i in range(self.X.shape[0]):
                Xi = self.X.values[i,0:4]
                predicted_result = self.X.values[i,4]
                a = np.dot(Xi,self.W) + self.bias
                if (predicted_result * a) <= 0:
                    self.new_weights(predicted_result,Xi)
                else:
                    pass

    # uses trained weights to test new data
    def test(self,Y):
        Xi = Y[0:4]
        a = np.dot(Xi,self.W) + self.bias
        if a > 0:
            return 1
        else:
            return 0
    
    ## uses trained weights to test new data and return a value for one vs rest confidence voting
    def multi_test(self,Y):
        Xi = Y[0:4]
        a = np.dot(Xi,self.W) + self.bias
        return a
                
class Perceptron_Regularisation(Perceptron) :
     
    def __init__(self,input,weights,bias,n_iterations,L):
        super().__init__(input, weights, bias, n_iterations)
        self.L = L
    
    # new_weights updates weights and bias
    def new_weights(self,predicted_result,Xi):
        for i in range(len(self.W)):
            self.W[i] = np.dot((1-2*self.L),self.W[i] + (Xi[i] *  predicted_result))
        self.bias = self.bias + predicted_result

#--------------------------------------------------------------------------------------------------------------------------------------
#Creating datasets for classification between 2 classes for one vs one approach
def preprocess_data(data_file, class_labels):
    df = pd.read_csv(data_file, header=None)
    # Filter rows where the 5th column is in class_labels
    df = df[(df[4] == class_labels[0]) | (df[4] == class_labels[1])]
    # Map classes to 1.0 and -1.0 in the 5th column
    df[4] = df[4].map({class_labels[0]: 1.0, class_labels[1]: -1.0})
    return df

# Specify the class labels for each case
class_labels_1vs2 = ['class-1', 'class-2']
class_labels_2vs3 = ['class-2', 'class-3']
class_labels_1vs3 = ['class-1', 'class-3']

# Process the data for each case
train_1vs2 = preprocess_data("data\\train.data", class_labels_1vs2)
train_2vs3 = preprocess_data("data\\train.data", class_labels_2vs3)
train_1vs3 = preprocess_data("data\\train.data", class_labels_1vs3)
test = pd.read_csv("data\\test.data",header=None)

#--------------------------------------------------------------------------------------------------------------------------------------
#Creating datasets for classification of each class one vs rest approach
def preprocess_multiclass_data(data_file, class_labels):
    df = pd.read_csv(data_file, header=None)
    # Map classes to 1.0 and -1.0 in the 5th column
    df[4] = df[4] = (df[4] == class_labels[0]).astype(int) * 2 - 1
    return df

# Specify the classes for each multiclass scenario
class_labels_1vsRest = ['class-1']
class_labels_2vsRest = ['class-2']
class_labels_3vsRest = ['class-3']

# Preprocess the data for each multiclass scenario
train_1vsRest = preprocess_multiclass_data("data\\train.data", class_labels_1vsRest)
train_2vsRest = preprocess_multiclass_data("data\\train.data", class_labels_2vsRest)
train_3vsRest = preprocess_multiclass_data("data\\train.data", class_labels_3vsRest)

#--------------------------------------------------------------------------------------------------------------------------------------
# Initialising classes objects for testing code and calling respective class methods for all classifications in order
# for multiclass classification in One vs One approach.
A = Perceptron(train_1vs2,[0,0,0,0],0,20)
A.train()
B = Perceptron(train_2vs3,[0,0,0,0],0,20)
B.train()
C = Perceptron(train_1vs3,[0,0,0,0],0,20)
C.train()

#Getting test accuracy
accurate_predictions = 0
for index, row in test.iterrows():
    class1_votes = 0
    class2_votes = 0
    class3_votes = 0
    original_class = row[4]
    result = A.test(row)
    if result == 1:
        class1_votes += 1
    else:
        class2_votes += 1
    result = B.test(row)
    if result == 1:
        class2_votes += 1
    else:
        class3_votes += 1
    result = C.test(row)
    if result == 1:
        class1_votes += 1
    else:
        class3_votes += 1

    if class1_votes > class2_votes and class1_votes > class3_votes:
        predicted_class = 'class-1'
    elif class2_votes > class1_votes and class2_votes > class3_votes:
        predicted_class = 'class-2'
    else:
        predicted_class = 'class-3'
    print(f"Predicted Class: {predicted_class}")
    if predicted_class == original_class:
        accurate_predictions += 1
print(f"Accuracy of one vs one approach to classify the data is {accurate_predictions/test.shape[0]*100}")

#--------------------------------------------------------------------------------------------------------------------------------------
# Initialising classes objects for testing code and calling respective class methods for all classifications in order
# for multiclass classification in One vs Rest approach.
A = Perceptron(train_1vsRest,[0,0,0,0],0,20)
A.train()
B = Perceptron(train_2vsRest,[0,0,0,0],0,20)
B.train()
C = Perceptron(train_3vsRest,[0,0,0,0],0,20)
C.train()

#Getting test accuracy
accurate_predictions = 0
for index, row in test.iterrows():
    class1_score = A.multi_test(row)
    class2_score = B.multi_test(row)
    class3_score = C.multi_test(row)
    original_class = row[4]

    if class1_score > class2_score and class1_score > class3_score:
        predicted_class = 'class-1'
    elif class2_score > class1_score and class2_score > class3_score:
        predicted_class = 'class-2'
    else:
        predicted_class = 'class-3'
    print(f"Predicted Class: {predicted_class}, Original Class: {original_class}")
    if predicted_class == original_class:
        accurate_predictions += 1
print(f"Accuracy of one vs Rest approach to classify the data is {accurate_predictions/test.shape[0]*100}")

#--------------------------------------------------------------------------------------------------------------------------------------
# Initialising classes objects for testing code and calling respective class methods for all classifications in order
# for multiclass classification in One vs Rest approach.
Regularisation_Coefficient = [.01, 0.1, 1.0, 10.0, 100.0]
for L in Regularisation_Coefficient:
    A = Perceptron_Regularisation(train_1vsRest,[0,0,0,0],0,20,L)
    A.train()
    B = Perceptron_Regularisation(train_2vsRest,[0,0,0,0],0,20,L)
    B.train()
    C = Perceptron_Regularisation(train_3vsRest,[0,0,0,0],0,20,L)
    C.train()

    #Getting test accuracy
    accurate_predictions = 0
    for index, row in test.iterrows():
        class1_score = A.multi_test(row)
        class2_score = B.multi_test(row)
        class3_score = C.multi_test(row)
        original_class = row[4]

        if class1_score > class2_score and class1_score > class3_score:
            predicted_class = 'class-1'
        elif class2_score > class1_score and class2_score > class3_score:
            predicted_class = 'class-2'
        else:
            predicted_class = 'class-3'
        print(f"Predicted Class: {predicted_class}, Original Class: {original_class}")
        if predicted_class == original_class:
            accurate_predictions += 1
    print(f"Accuracy of one vs Rest with Regularisation Coefficient {L} to classify the data is {accurate_predictions/test.shape[0]*100}")