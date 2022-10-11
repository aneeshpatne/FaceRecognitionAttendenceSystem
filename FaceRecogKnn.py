import cv2,time
import pickle
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import warnings
import os
import pandas as pd
from datetime import datetime
warnings.filterwarnings("ignore")
def record_attendance(name, direction):
    if 'attendance.csv' not in os.listdir():
        attendance = pd.DataFrame(columns=['Name','Date','Time','Entry/Exit'])
        attendance.to_csv('attendance.csv')

    attendance_log = pd.read_csv('attendance.csv', index_col=[0])
    date = datetime.now()

    time = date.strftime('%X')
    day = f'{date.day}/{date.month}/{date.year}'

    attendance = pd.DataFrame([[name, day, time, direction]], columns=['Name','Date','Time','Entry/Exit'])
    attendance_log = pd.concat([attendance_log, attendance], ignore_index=True)

    attendance_log.to_csv('attendance.csv')
def predict(img_path, knn_clf=None, model_path=None, threshold=0.6): # 6 needs 40+ accuracy, 4 needs 60+ accuracy
    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")
    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)
    # Load image file and find face locations
    img = img_path
    face_box = face_recognition.face_locations(img)
    # If no faces are found in the image, return an empty result.
    if len(face_box) == 0:
        return []
    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(img, known_face_locations=face_box)
    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=2)
    matches = [closest_distances[0][i][0] <= threshold for i in range(len(face_box))]
    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings),face_box,matches
    )]
webcam = cv2.VideoCapture(0) #  0 to use webcam 
while True:
    # Loop until the camera is working
    rval = False
    while(not rval):
        # Put the image from the webcam into 'frame'
        (rval, frame) = webcam.read()
        if(not rval):
            print("Failed to open webcam. Trying again...")
            
    # Flip the image (optional)
    frame=cv2.flip(frame,1) # 0 = horizontal ,1 = vertical , -1 = both
    frame_copy = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    frame_copy=cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
    predictions = predict(frame_copy, model_path="classifier/trained_knn_model.clf") 
    if len(predictions)==0:
        print("NOT FACE DETECTED")
    else:    
        print(predictions[0][0])
        name = predictions[0][0] # add path here
        if name == 'unknown':
            print('Captured face not in database...')
            print('Please register and try again!')
            break
        else:
            print('1. Entry time')
            print('2. Exit time')
            direction_input = int(input('Select option to record Entry/Exit time: '))
            direction = ''

            if direction_input == 1:
                direction = 'Entry'
            elif direction_input == 2:
                direction = 'Exit'
            else:
                print('\nInvalid input\n')
        print('Closing application...')

        if direction_input == 1 or direction_input == 2:
            print('face detected...')
            print(f'Name: {name}')
            confirmation = input(f'Are you sure you want to continue recording attendance for {name}?(y/n): ')
            if confirmation == 'y' or confirmation == 'Y':
                record_attendance(name, direction)
                print(f'\n{direction} time has been recorded for {name}...')
            elif confirmation == 'n' or confirmation == 'N':
                print('Please try face capture again!')
            else:
                print('\nInvalid input\n')
                print('Closing application...')

    #p rint(predictions[0][0])
    font = cv2.FONT_HERSHEY_DUPLEX
    for name, (top, right, bottom, left) in predictions:
        top *= 4 #scale back the frame since it was scaled to 1/4 in size
        right *= 4
        bottom *= 4
        left *= 4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 4)
        cv2.putText(frame, name, (left-10,top-6), font, 2, (0, 0, 255), 2)
    
        
    #print(name)
        
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
webcam.release()
cv2.destroyAllWindows()

