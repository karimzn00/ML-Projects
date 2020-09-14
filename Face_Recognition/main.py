import cv2
from imutils import paths
from imutils.video import VideoStream, FPS
import os 
import  imutils
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from keras.layers import Dense
from tensorflow import convert_to_tensor
from keras import models 
import time
import keras
import argparse
import pandas as pd
import sklearn.utils import shuffle




ar = argparse.ArgumentParser()
ar.add_argument('-c', '--choice', required = True)
arg = vars(ar.parse_args())

# load the detector from caffe and torch
detector = cv2.dnn.readNetFromCaffe('detector/deploy.prototxt', 'detector/res10_300x300_ssd_iter_140000.caffemodel')

#load the embeddings model 
embedder = cv2.dnn.readNetFromTorch('openface_nn4.small2.v1.t7')

# select dataset folder
imagePaths = list(paths.list_images('dataset'))

def read(image, scaller, swap):
    image = imutils.resize(image, width = 600)
    (h, w) = image.shape[:2]
    ib = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), scaller, swapRB=swap, crop=False)
    detector.setInput(ib)
    detections = detector.forward()

    return [detections, (h, w)]


def Embedding(imagePaths):
    total = 0
    embeddings = []
    persons = []
    for (i, imagePath) in enumerate(imagePaths):
        print('[KIMTRON] Encoding Image : {}/{}'.format(i+1, len(imagePaths)))

        name = imagePath.split(os.path.sep)[-2]
        image = cv2.imread(imagePath)
        
        detections = read(image, (104.4, 177.8, 123.0), False)[0]
        (h, w)= read(image, (104.4, 177.8, 123.0), False)[1]

        if len(detections) > 0:
            i = np.argmax(detections[0, 0, :, 2])   
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:     
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (sX, sY, eX, eY) = box.astype('int')

                face = image[sY:eY, sX:eX]
                (fH, fW) = face.shape[:2]

                if fW < 20 or fH < 20:
                    continue 
                
                fb = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop= False)
                embedder.setInput(fb)
                vec = embedder.forward()

                persons.append(name)
                embeddings.append(vec.flatten()) 
                total += 1 
    data = {'embeddings' : embeddings, 'name' : persons}
    print('[KIMTRON] encodings : {} ...'.format(total))

    with open('embeddings', 'wb') as f:
        pickle.dump(data, f)
        f.close 



def train():
    data = pickle.loads(open('embeddings', 'rb').read())
    y = pd.DataFrame(data['name'], columns = ['names'])
    lab = LabelEncoder()
    y['cat'] = lab.fit_transform(y['names'])
    y = shuffle(y)
    output_lent = 1

    x = data['embeddings']
    x = convert_to_tensor(x)
    model = Sequential()
    model.add(Dense(50, input_dim = 128 , activation='relu'))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(500, activation = 'relu'))
    model.add(Dense(1000, activation = 'relu'))
    model.add(Dense(1500, activation = 'relu'))
    model.add(Dense(500, activation = 'relu'))
    model.add(Dense(output_lent, activation = 'softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer ='adam', metrics = ['accuracy'])

    model.fit(x, y['cat'], epochs=1000, batch_size = 10, validation_split=0.25)
    model.save('modelNN')       
   
    with open('persons', 'wb') as f:
        pickle.dump(y, f)
        f.close()

def stream(src):

    vs = VideoStream(src=src).start()
    time.sleep(2.0)
    fps = FPS().start()

    model = models.load_model('modelNN')
    #model = pickle.load(open('cnn', 'rb'))
    persons = pickle.loads(open('persons', 'rb').read())


    
    print('[KIMTRON] Streaming ... ')





    while True:
        frame = vs.read()
        detections = read(frame, (104.0, 177.0, 123.0), False)[0]
        (h, w) = read(frame, (104.0, 177.0, 123.0), False)[1]
        
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (sX, sY, eX, eY) = box.astype('int')
                face = frame[sY:eY, sX:eX]
                (fH, wH) = face.shape[:2]
                if fH < 20 or wH < 20:
                    continue

                fb = cv2.dnn.blobFromImage(face, 1.0, (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(fb)
                vec = embedder.forward()
                vec =  convert_to_tensor(vec)
                preds = model.predict(vec) 
                
             
                j = np.argmax(preds)

                names = persons['names']
                name = names[persons['cat']==j][0]
                text = str(name)
                y = sY - 10 if sY -10 > 10 else sY +10
                x = sX - 10 if sX -10 > 10 else sX +10

                cv2.rectangle(frame, (sX , sY), (eX, eY), (0, 0, 255), 1)
                cv2.putText(frame, text, (sX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                
            fps.update()
        cv2.imshow("KIMTRON", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

    fps.stop()


################################################################################################################################################


c = arg['choice']




if c == '1':
    Embedding(imagePaths)


if c == '2':
    train()

if c == '3':
    stream(0)
    vs.stop()
    cv2.destroyAllWindows()
