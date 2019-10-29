import dlib
import cv2
import numpy as np
import glob
import random
import math
#import itertools
from sklearn.svm import SVC
import pickle
import os
from sklearn.metrics import classification_report, confusion_matrix

#Emotion list
emotions = ["anger", "disgust", "happy", "neutral", "surprise"]
detector = dlib.get_frontal_face_detector()
model = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
clf = SVC(C=0.3, kernel='poly',degree=3 , decision_function_shape='ovo', probability=True)   #Set the classifier as a support vector machines with linear kernel

def get_files(emotion):
    images = glob.glob("dataset_combine\\%s\\*" %emotion)    
    random.shuffle(images)
    training_set = images[:int(len(images)*0.8)]   #get 80% of image files to be trained
    testing_set = images[-int(len(images)*0.2):]   #get 20% of image files to be tested
    return training_set, testing_set

def get_landmarks(image):
    detections = detector(image, 1)
    #For all detected face instances individually
    for k,d in enumerate(detections):
        #get facial landmarks with prediction model
        shape = model(image, d)
        xpoint = []
        ypoint = []
        for i in range(17, 68):
            xpoint.append(float(shape.part(i).x))
            ypoint.append(float(shape.part(i).y))

        #center points of both axis
        xcenter = np.mean(xpoint)
        ycenter = np.mean(ypoint)
        #Calculate distance between particular points and center point
        xdistcent = [(x-xcenter) for x in xpoint]
        ydistcent = [(y-ycenter) for y in ypoint]

        #prevent divided by 0 value
        if xpoint[31] == xpoint[34]:
            angle_nose = 0    #31,34?
        else:
            #point 14 is the tip of the nose, point 11 is the top of the nose brigde
            angle_nose = int(math.atan((ypoint[31]-ypoint[34])/(xpoint[31]-xpoint[34]))*180/math.pi)

        #Get offset by finding how the nose brigde should be rotated to become perpendicular to the horizontal plane
        if angle_nose < 0:
            angle_nose += 90
        else:
            angle_nose -= 90

        landmarks = []
        for cx,cy,x,y in zip(xdistcent, ydistcent, xpoint, ypoint):
            #Add the coordinates relative to the centre of gravity
            landmarks.append(cx)
            landmarks.append(cy)

            #Get the euclidean distance between each point and the centre point (the vector length)
            meanar = np.asarray((ycenter,xcenter))
            centpar = np.asarray((y,x))
            dist = np.linalg.norm(centpar-meanar)

            #Get the angle the vector describes relative to the image, corrected for the offset that the nosebrigde has when the face is not perfectly horizontal
            if x == xcenter:
                angle_relative = 0
            else:
                angle_relative = (math.atan(float(y-ycenter)/(x-xcenter))*180/math.pi) - angle_nose
                #print(anglerelative)
            landmarks.append(dist)
            landmarks.append(angle_relative)

    if len(detections) < 1:
        #In case no case selected, print "error" values
        landmarks = "error"
    
    return landmarks
'''
def random_subsampling(dataset,turn,length):                           #定义随机过采样函数
    new_dataset = []                                                   #初始化
    for t in range(0,turn):                                            #给定过采样轮次
        index = np.arange(0,len(dataset))                              #获取列表索引
        np.random.shuffle(index)                                       #打乱列表索引
        for s in range(0,length):                                      #给定过采样数据规模
           new_dataset.append(dataset[index[s]])                       #过采样元素添加到新列表
    return new_dataset
'''
def make_sets():
    training_data = []
    training_label = []
    testing_data = []
    testing_label = []
    for emotion in emotions:
        training_set, testing_set = get_files(emotion)
        #add data to training and testing dataset, and generate labels 0-4
        for item in training_set:
            #read image
            img = cv2.imread(item)
            #convert to grayscale
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe_img = clahe.apply(gray_img)
            landmarks_vec = get_landmarks(clahe_img)

            if landmarks_vec == "error":
                pass
            else:
                training_data.append(landmarks_vec)
                training_label.append(emotions.index(emotion))

        for item in testing_set:
            img = cv2.imread(item)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe_img = clahe.apply(gray_img)
            landmarks_vec = get_landmarks(clahe_img)
            if landmarks_vec == "error":
                pass
            else:
                testing_data.append(landmarks_vec)
                testing_label.append(emotions.index(emotion))

    return training_data, training_label, testing_data, testing_label

def create_model():
    accur_lin = []
    max_accur = 0
    for i in range(0,5):
        #Make sets by random sampling 80/20%
        print("Marking set %s" %i)
        X_train, y_train, X_test, y_test = make_sets()
        
        #Turn the training set into a numpy array for the classifier
        np_train = np.array(X_train)
        np_test = np.array(y_train)
        #train SVM
        print("Trainging SVM Classifier %s" %i)
        clf.fit(np_train, np_test)

        #Use score() function to get accuracy
        print("Getting accuracy score -- %s" %i)
        #npar_pred = np.array(X_test)
        pred_lin = clf.score(X_test, y_test)

        #Find Best Accuracy and save to file
        if pred_lin > max_accur:
            max_accur = pred_lin
            max_clf = clf

        print("Test Accuracy: ", pred_lin)
        accur_lin.append(pred_lin)  #Store accuracy in a list

    print("Mean Accuracy Value: %.3f" %np.mean(accur_lin))   #Get mean accuracy of the 10 runs

    predictions = max_clf.predict(X_test)
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))

    return max_accur, max_clf

if __name__ == '__main__':
    max_accur, max_clf = create_model()
    print('Best accuracy = ', max_accur*100, 'percent')
    print(max_clf)
    try:
        os.remove('models\model1.pkl')
    except OSError:
        pass
    output = open('models\model1.pkl', 'wb')
    pickle.dump(max_clf, output)
    output.close()
