import tensorflow as tf
from tensorflow.keras import layers, models
import cv2
import os
import numpy as np
import random

th1=30
th2=55
size=42
video_dir="server-ia/data/videos/"
dir_images_panneaux="server-ia/data/images/panneaux"

def panneau_model(nbr_classes):
    model=tf.keras.Sequential()

    model.add(layers.Input(shape=(size, size, 3), dtype='float32'))
    
    model.add(layers.Conv2D(128, 3, strides=1))
    model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    model.add(layers.Conv2D(128, 3, strides=1))
    model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    model.add(layers.MaxPool2D(pool_size=2, strides=2))

    model.add(layers.Conv2D(256, 3, strides=1))
    model.add(layers.Dropout(0.3))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    model.add(layers.Conv2D(256, 3, strides=1))
    model.add(layers.Dropout(0.4))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    model.add(layers.MaxPool2D(pool_size=2, strides=2))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(nbr_classes, activation='sigmoid'))
    
    return model

def lire_images_panneaux(dir_images_panneaux, size=None):
    tab_panneau=[]
    tab_image_panneau=[]

    if not os.path.exists(dir_images_panneaux):
        quit("Le repertoire d'image n'existe pas: {}".format(dir_images_panneaux))

    files=os.listdir(dir_images_panneaux)
    if files is None:
        quit("Le repertoire d'image est vide: {}".format(dir_images_panneaux))

    for file in sorted(files):
        if file.endswith("png"):
            tab_panneau.append(file.split(".")[0])
            image=cv2.imread(dir_images_panneaux+"/"+file)
            if size is not None:
                image=cv2.resize(image, (size, size), cv2.INTER_LANCZOS4)
            tab_image_panneau.append(image)
            
    return tab_panneau, tab_image_panneau

tab_panneau, tab_image_panneau=lire_images_panneaux(dir_images_panneaux)

model_panneau=panneau_model(len(tab_panneau))
checkpoint=tf.train.Checkpoint(model_panneau=model_panneau)
checkpoint.restore(tf.train.latest_checkpoint("server-ia\data\modeles\road_sign_speed_trainers/"))

l=os.listdir(video_dir)
random.shuffle(l)

for video in l:
    if not video.endswith("mp4"):
        continue
    cap=cv2.VideoCapture(video_dir+"/"+video)

    print("video:", video)
    id_panneau=-1
    while True:
        ret, frame=cap.read()
        if ret is False:
            break
        f_w, f_h, f_c=frame.shape
        frame=cv2.resize(frame, (int(f_h/1.5), int(f_w/1.5)))

        image=frame[200:400, 700:1000]

        # represents the top left corner of rectangle 
        start_point = (600, 50)
        # Ending coordinate
        # represents t
        # he bottom right corner of rectangle 
        end_point = (800, 450)
        # Color in BGR 
        color = (255, 255, 255)
        # Line thickness
        thickness = 1

        cv2.rectangle(frame, start_point, end_point, color, thickness) 

        gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        circles=cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=th1, param2=th2, minRadius=5, maxRadius=45)
        if circles is not None:
            circles=np.int16(np.around(circles))
            for i in circles[0,:]:
                if i[2]!=0:
                    panneau=cv2.resize(image[max(0, i[1]-i[2]):i[1]+i[2], max(0, i[0]-i[2]):i[0]+i[2]], (size, size))/255
                    cv2.imshow("panneau", panneau)
                    prediction=model_panneau(np.array([panneau]), training=False)
                    print("Prediction:", prediction)
                    if np.any(np.greater(prediction[0], 0.6)):
                        id_panneau=np.argmax(prediction[0])
                        print("   -> C'est un panneau:", tab_panneau[id_panneau], "KM/H")
                        w, h, c=tab_image_panneau[id_panneau].shape
                    else:
                        print("   -> Ce n'est pas un panneau")
        if id_panneau!=-1:
            frame[0:h, 0:w, :]=tab_image_panneau[id_panneau]
        cv2.putText(frame, "fichier:"+video, (30, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow("Video", frame)
        key=cv2.waitKey(1)&0xFF
        if key==ord('q'):
            quit()
        if key==ord('a'):
            for cpt in range(100):
                ret, frame=cap.read()
        if key==ord('f'):
            break

cv2.destroyAllWindows()
