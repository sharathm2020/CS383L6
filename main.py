import numpy as np
import pandas as pd
from knn import evaluate_model
from PIL import Image
import os


#Function to process the ctg.csv dataset
def process_ctg_data(filepath):
    #Read in the data and drop the class column
    data = pd.read_csv(filepath)
    data = data.drop(columns=['CLASS'])
    
    #Seed and shuffle the data randomly
    np.random.seed(0)
    data = data.values
    np.random.shuffle(data)
    data = pd.DataFrame(data)

    #Split the data into training and validation sets, 2/3 training, 1/3 validation
    split_index = int(np.ceil(2/3 * len(data)))
    train_data = data.iloc[:split_index]
    val_data = data.iloc[split_index:]

    #Split the data into features and labels for both training and validation sets
    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values
    X_val = val_data.iloc[:, :-1].values
    y_val = val_data.iloc[:, -1].values

    #Printed out the predictions and validation labels and received
    #an NaN value error for a value within the validation set, so I added in filtering for NaN values
    valid_indices = ~np.isnan(y_val)
    X_val = X_val[valid_indices]
    y_val = y_val[valid_indices]
    
    return X_train, y_train, X_val, y_val


#Function to process the yale faces dataset
def process_yale_faces(directory_path):
    image_dir = directory_path
    X = []
    labels = []
    
    
    for filename in os.listdir(image_dir):
        #Skip the readme.txt file
        if filename == 'Readme.txt':
            continue
            
        #Process the files that begin with "subject"
        if filename.startswith('subject'):
            try:
                #Extract the subject number from the filename and convert it to an integer
                subject_part = filename.split('.')[0]
                label = int(subject_part.replace('subject', ''))
                
                #Open the image, resize and flatten it
                img = Image.open(os.path.join(image_dir, filename)).convert('L')
                img_resized = img.resize((40, 40))
                img_array = np.array(img_resized).flatten()
                
                X.append(img_array)
                labels.append(label)
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    X = np.array(X)
    labels = np.array(labels)
    
    #Find any unique labels, create empty lists for training and validation sets
    unique_labels = np.unique(labels)
    X_train, y_train, X_val, y_val = [], [], [], []
    
    #Iterate over each unique label found, shuffle any indices found for that label
    #and split the data into training and validation sets
    for label in unique_labels:
        indices = np.where(labels == label)[0]
        np.random.shuffle(indices)
        split_index = int(np.ceil(2/3 * len(indices)))
        X_train.extend(X[indices[:split_index]])
        y_train.extend(labels[indices[:split_index]])
        X_val.extend(X[indices[split_index:]])
        y_val.extend(labels[indices[split_index:]])
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    
    
    return X_train, y_train, X_val, y_val

#Main Function
def main():
    #Call Process Data function for ctg dataset
    #Set k to 3 for kNN algorithm
    #Evaluate the model for accuracy and confusion matrix
    #Print results
    X_train, y_train, X_val, y_val = process_ctg_data('CTG.csv')
    k = 3
    accuracy, confusion_matrix = evaluate_model(X_train, y_train, X_val, y_val, k)
    print(f'CTG Dataset - Validation Accuracy: {accuracy}')
    print('Confusion Matrix:')
    print(confusion_matrix)

    #Call process data function for yale faces dataset
    #K is still set to 3 from before
    #Evaluate the model for accuracy and confusion matrix
    #Print Results
    X_train, y_train, X_val, y_val = process_yale_faces('yalefaces')
    print(f'Yale Faces - Training set size: {len(X_train)}, Validation set size: {len(X_val)}')
    accuracy, confusion_matrix = evaluate_model(X_train, y_train, X_val, y_val, k)
    print(f'Yale Faces Dataset - Validation Accuracy: {accuracy}')
    print('Confusion Matrix:')
    print(confusion_matrix)

if __name__ == "__main__":
    main() 