import numpy as np
import pandas as pd

#Function to calculate the euclidean distance between two points
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

#Function to get the k nearest neighbors to our given target point
def get_neighbors(X_train, y_train, test_instance, k):
    distances = []
    #Iterate over training set
    for i in range(len(X_train)):
        #Calculate the distance between the training point and the target
        dist = euclidean_distance(X_train[i], test_instance)
        distances.append((y_train[i], dist))
    #Sort the distances list in ascending order
    distances.sort(key=lambda x: x[1])
    #Get the k nearest neighbors
    neighbors = distances[:k]
    #Return the labels of the k nearest neighbors
    return [neighbor[0] for neighbor in neighbors]

#Function for predicting the classification of the target point
def predict_classification(X_train, y_train, test_instance, k):
    #Get k nearest neighbors to the target
    neighbors = get_neighbors(X_train, y_train, test_instance, k)
    class_count = {}
    #For each neighbor instance, increment the count for the class
    for neighbor in neighbors:
        if neighbor in class_count:
            class_count[neighbor] += 1
        else:
            class_count[neighbor] = 1
    #Return the class with the highest count
    if class_count:
        return max(class_count, key=class_count.get)
    else:
        return None

#Function to evaluate the model
def evaluate_model(X_train, y_train, X_val, y_val, k):
    #Predict the classification of each validation point
    predictions = [predict_classification(X_train, y_train, x, k) for x in X_val]
    predictions = np.array(predictions)

    #Filter out None predictions(Code was added in as a fix for the Nan value error)
    valid_indices = [i for i, pred in enumerate(predictions) if pred is not None]
    predictions = predictions[valid_indices]
    y_val = y_val[valid_indices]

    #Calculate the accuracy of the model
    accuracy = np.mean(predictions == y_val)
    
    #Find the minimum and maximum class labels
    min_class = int(min(min(y_val), min(predictions)))
    max_class = int(max(max(y_val), max(predictions)))
    
    #Create a confusion matrix with the correct size
    size = max_class - min_class + 1
    confusion_matrix = np.zeros((size, size), dtype=int)
    
    #Fill the confusion matrix with the correct values
    for i in range(len(y_val)):
        true_label = int(y_val[i]) - min_class
        pred_label = int(predictions[i]) - min_class
        confusion_matrix[true_label][pred_label] += 1
    
    return accuracy, confusion_matrix 