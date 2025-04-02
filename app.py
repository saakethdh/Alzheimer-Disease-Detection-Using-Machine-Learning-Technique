import shutil
import os
import sys
from flask import Flask, render_template, request
import cv2  # working with, mainly resizing, images
import numpy as np  # dealing with arrays
import os  # dealing with directories
from random import shuffle  # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import \
    tqdm  # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
import matplotlib.pyplot as plt



# global b
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('home.html')

@app.route('/image', methods=['GET', 'POST'])
def image():
    if request.method == 'POST':
        
                
        dirPath = "static/images"
        fileList = os.listdir(dirPath)
        for fileName in fileList:
            os.remove(dirPath + "/" + fileName)
        fileName=request.form['filename']
        dst = "static/images"
        
        shutil.copy("test/"+fileName, dst)

        
        verify_dir = 'static/images'
        IMG_SIZE = 50
        LR = 1e-3
        MODEL_NAME = 'Alzheimers-{}-{}.model'.format(LR, '2conv-basic')
    
        def process_verify_data():
            verifying_data = []
            for img in os.listdir(verify_dir):
                path = os.path.join(verify_dir, img)
                img_num = img.split('.')[0]
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                verifying_data.append([np.array(img), img_num])
                np.save('verify_data.npy', verifying_data)
            return verifying_data

        verify_data = process_verify_data()
        #verify_data = np.load('verify_data.npy')

        
        tf.compat.v1.reset_default_graph()
        #tf.reset_default_graph()

        convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 128, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, 0.8)

        convnet = fully_connected(convnet, 4, activation='softmax')
        convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

        model = tflearn.DNN(convnet, tensorboard_dir='log')

        if os.path.exists('{}.meta'.format(MODEL_NAME)):
            model.load(MODEL_NAME)
            print('model loaded!')


        fig = plt.figure()
        accuracy=" "
        str_label=" "
        for num, data in enumerate(verify_data):

            img_num = data[1]
            img_data = data[0]

            y = fig.add_subplot(3, 4, num + 1)
            orig = img_data
            data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
            # model_out = model.predict([data])[0]
            model_out = model.predict([data])[0]
            print(model_out)
            print('model {}'.format(np.argmax(model_out)))

            if np.argmax(model_out) == 0:
                str_label = 'MildDemented'
            elif np.argmax(model_out) == 1:
                str_label = 'ModerateDemented'
            elif np.argmax(model_out) == 2:
                str_label = 'NonDemented'
            elif np.argmax(model_out) == 3:
                str_label = 'VeryMildDemented'
           

            if str_label == 'MildDemented':
                status = "MildDemented "
                print("The predicted image of the MildDemented is with a accuracy of {} %".format(model_out[0]*100))
                accuracy = "The predicted image of the MildDemented is with a accuracy of {} %".format(model_out[0]*100)
                A=float(model_out[0])
                B=float(model_out[1])
                C=float(model_out[2])
                D=float(model_out[3])
                dic={'MildDemented':A,'Moderate':B,'NonDemented':C,'VeryMild':D}
                algm = list(dic.keys()) 
                accu = list(dic.values()) 
                fig = plt.figure(figsize = (5, 5))  
                plt.bar(algm, accu, color ='maroon', width = 0.3)  
                plt.xlabel("Comparision") 
                plt.ylabel("Accuracy Level") 
                plt.title("Accuracy Comparision between covid detection....")
                plt.savefig('static/matrix.png')
                # plt.show()


              
            
            elif str_label == 'ModerateDemented':
                status = "ModerateDemented"
                print("The predicted image of the ModerateDemented is with a accuracy of {} %".format(model_out[1]*100))
                accuracy = "The predicted image of the ModerateDemented is with a accuracy of {} %".format(model_out[1]*100)
                A=float(model_out[0])
                B=float(model_out[1])
                C=float(model_out[2])
                D=float(model_out[3])
                dic={'MildDemented':A,'Moderate':B,'NonDemented':C,'VeryMild':D}
                algm = list(dic.keys()) 
                accu = list(dic.values()) 
                fig = plt.figure(figsize = (5, 5))  
                plt.bar(algm, accu, color ='maroon', width = 0.3)  
                plt.xlabel("Comparision") 
                plt.ylabel("Accuracy Level") 
                plt.title("Accuracy Comparision between Alzheimers detection....")
                plt.savefig('static/matrix.png')
                # plt.show()
                
            elif str_label == 'NonDemented':
                status = "NonDemented"
                print("The predicted image of the NonDemented is with a accuracy of {} %".format(model_out[2]*100))
                accuracy = "The predicted image of the NonDemented is with a accuracy of {} %".format(model_out[2]*100)
                A=float(model_out[0])
                B=float(model_out[1])
                C=float(model_out[2])
                D=float(model_out[3])
                dic={'MildDemented':A,'Moderate':B,'NonDemented':C,'VeryMild':D}
                algm = list(dic.keys()) 
                accu = list(dic.values()) 
                fig = plt.figure(figsize = (5, 5))  
                plt.bar(algm, accu, color ='maroon', width = 0.3)  
                plt.xlabel("Comparision") 
                plt.ylabel("Accuracy Level") 
                plt.title("Accuracy Comparision between Alzheimers detection....")
                plt.savefig('static/matrix.png')
                # plt.show()
                
                    
            elif str_label == 'VeryMildDemented':
                status= 'VeryMildDemented'
                print("The predicted image of the VeryMildDemented is with a accuracy of {} %".format(model_out[3]*100))
                accuracy = "The predicted image of the VeryMildDemented is with a accuracy of {} %".format(model_out[3]*100)
                A=float(model_out[0])
                B=float(model_out[1])
                C=float(model_out[2])
                D=float(model_out[3])
                dic={'MildDemented':A,'Moderate':B,'NonDemented':C,'VeryMild':D}
                algm = list(dic.keys()) 
                accu = list(dic.values()) 
                fig = plt.figure(figsize = (5, 5))  
                plt.bar(algm, accu, color ='maroon', width = 0.3)  
                plt.xlabel("Comparision") 
                plt.ylabel("Accuracy Level") 
                plt.title("Accuracy Comparision between Alzheimers detection....")
                plt.savefig('static/matrix.png')
                # plt.show()
                
           

            
        return render_template('home.html', status=str_label,accuracy=accuracy, ImageDisplay="http://127.0.0.1:5000/static/images/"+fileName,ImageDisplay1="http://127.0.0.1:5000/static/matrix.png")
    return render_template('home.html')
if __name__ == '__main__':
    app.run(debug=True)
