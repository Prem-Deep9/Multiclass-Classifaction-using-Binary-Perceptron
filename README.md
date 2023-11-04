Implementing Binary Perception from scratch using python.

# Perceptron

# Perceptron Algorithm

# Binary Perceptron

# One-vs-One Approach
In this approach a binary classifier is trained for every pair of classes. The idea is to classify the data into one of the two classes in each binary classifier. When you want to make a prediction, you apply all classifiers to the data point and use a voting mechanism (e.g., majority voting) to determine the final class. Here's how it works:
1. For each pair of classes i and,j, use training objects from classes i and j to train
algorithm A to distinguish between objects in classes i and j. Denote the obtained
classifier A<sub>ij</sub>
2. For K classes, we train k(k-1)/2 prediction models.
3. Applying the prediction model A<sub>ij</sub> to an incoming object $\bar{X}$ is interpreted as voting. A<sub>ij</sub> votes either +1 for $\bar{X}$ to be in class i, or A<sub>ij</sub> votes +1 for $\bar{X}$ to be in class j.
4. For an incoming object $\bar{X}$, apply all prediction models one by one.
5. The class label with the most votes is declared as the winner.

# One-vs-Many Approach

# Regularisation
