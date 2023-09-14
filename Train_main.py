import os
import math
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import neighbors
import face_recognition
from sklearn import svm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.naive_bayes import GaussianNB
from face_recognition.face_recognition_cli import image_files_in_folder
import pickle
from sklearn.decomposition import PCA as RandomizedPCA
def train(train_dir, model_save_path, n_neighbors=2, knn_algo='ball_tree', verbose=False):
    X = []
    y = []
    # Loop through each person in the training set
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue
        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)
            print( "processing :",img_path)
            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, skip the image.
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # Add face encoding for current image to the training set
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)

    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)
    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    from sklearn.model_selection import train_test_split
    features = np.array(y)
    train_features, test_features, train_labels, test_labels = train_test_split(X, features, test_size = 0.2,random_state = 5)
    knn_clf.fit(train_features, train_labels)
    #SVM
    clf_SVM = svm.SVC(gamma ='scale',kernel = 'linear', shrinking = False,probability=True)
    clf_SVM.fit(train_features, train_labels)
    #RANDOM FOREST
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(max_depth=2, random_state=0)
# Train the model on training data
    rf.fit(train_features, train_labels);
    #NAIVE_BAYES
    gnb = GaussianNB()
    gnb.fit(train_features, train_labels)




    predictions = knn_clf.predict(test_features)
    predictions_SVM = clf_SVM.predict(test_features)
    predictions_RF = rf.predict(test_features)
    predictions_NB = gnb.predict(test_features)
    print(predictions)
    print(test_labels)
    
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(predictions,test_labels)
    accuracy_SVM = accuracy_score(predictions_SVM,test_labels)
    accuracy_RF = accuracy_score(predictions_RF,test_labels)
    accuracy_NB = accuracy_score(predictions_NB,test_labels)
    print("TRAINING COMPLETE")
    print('KNN ACCURACY:',accuracy*100)
    print('SVM ACCURACY:',accuracy_SVM*100)
    print('RANDOM FOREST ACCURACY:',accuracy_RF*100)
    print('NAIVE BAYES ACCURACY:',accuracy_NB*100)

    cm1 = confusion_matrix(predictions, test_labels)
    cm2 = confusion_matrix(predictions_SVM, test_labels)
    cm3 = confusion_matrix(predictions_RF, test_labels)
    cm4 = confusion_matrix(predictions_NB, test_labels)
    #disp = ConfusionMatrixDisplay(confusion_matrix=cm1,display_labels=['macron','modi','aneesh','sakpal','chris_evans','mark_ruffalo','chris_hemsworth','robert_downey_jr','scarlett_johansson'])
   # disp1 = ConfusionMatrixDisplay(confusion_matrix=cm2,display_labels=['macron','modi','aneesh','sakpal','chris_evans','mark_ruffalo','chris_hemsworth','robert_downey_jr','scarlett_johansson'])
    #disp2 = ConfusionMatrixDisplay(confusion_matrix=cm3,display_labels=['macron','modi','aneesh','sakpal','chris_evans','mark_ruffalo','chris_hemsworth','robert_downey_jr','scarlett_johansson'])
    #disp3 = ConfusionMatrixDisplay(confusion_matrix=cm4,display_labels=['macron','modi','aneesh','sakpal','chris_evans','mark_ruffalo','chris_hemsworth','robert_downey_jr','scarlett_johansson'])
    #disp.plot()
    #disp1.plot()
    #disp2.plot()
   # disp3.plot()
   # plt.show()

    # Save the trained KNN classifier

    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)
        print("Training completed now")
    return knn_clf
train("train_image","classifier/trained_knn_model.clf") # add path here
