# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 11:15:50 2019

@author: Ajay Solanki
"""

from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras.models import Model
from keras import models
from keras import optimizers
from keras.preprocessing import image
import keras
import os,shutil
import pandas as pd
from keras.applications import Xception
import numpy as np
import random
import matplotlib.pyplot as plt


from keras import Input

class ImageComparator:
    

    def copy_files(self, imagelist_file_name,image_dir,train_dir,image_files_directory):
        imageslist_file = os.path.join(image_dir, imagelist_file_name)
        data = pd.read_csv(imageslist_file, sep=" ", header=None)
        data.columns = ["filename", "category"]
        
        for ind in data.index:
            fname = data["filename"][ind]
            src = os.path.join(image_dir, fname)
            dstfile_name =  fname.strip("calibrated/")
            category_name = data["category"][ind].astype(str)
            #print(category_name)
            dst_dir = os.path.join(train_dir,category_name  )
            if not os.path.exists(dst_dir):
                os.mkdir(dst_dir)
            dst = os.path.join(dst_dir, dstfile_name)
            shutil.copyfile(src, dst)
    
    # Generate Copy Images from the Data
    def populate_data_for_comparator(self, imagelist_file_name,image_dir,train_dir,image_files_directory):
      
        
        imageslist_file = os.path.join(image_dir, imagelist_file_name)
        data = pd.read_csv(imageslist_file, sep=" ", header=None)
        datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')
        data.columns = ["filename", "category"]
        print (data.shape[0])
        
        X_left = [] #np.empty(data.count, 255,255,3) #
        X_right = [] #
        y = [] #
        match_count = 0
        row_count = data.shape[0]
        for ind in data.index:
            fname = data["filename"][ind]
            src = os.path.join(image_dir, fname)
            dstfile_name =  fname.strip("calibrated/")
            category_name = data["category"][ind].astype(str)
            #print(category_name)
            dst_dir = os.path.join(train_dir,category_name  )
            if not os.path.exists(dst_dir):
                os.mkdir(dst_dir)
            dst = os.path.join(dst_dir, dstfile_name)
            img = image.load_img(dst, target_size=(255, 255))
            x = image.img_to_array(img)
            X_left.append(x)
            
            
            flag = 0
            if (ind % 6) and (match_count < row_count * 0.2) :
                # This is path for matched images
                x = x.reshape((1,) + x.shape)
                i = 0
                for batch in datagen.flow(x, batch_size=1):
                    X_right.append(batch[0])
                    flag=1
                    match_count = match_count + 1
                    break
            else :
                r_index = random.randint(0, row_count-1)
                #print(r_index)
                fname = data["filename"][r_index]
                category_name = data["category"][r_index].astype(str)
                dstfile_name =  fname.strip("calibrated/")
                dst_dir = os.path.join(train_dir,category_name  )
                dst = os.path.join(dst_dir, dstfile_name)
                img = image.load_img(dst, target_size=(255, 255))
                x = image.img_to_array(img)
                X_right.append(x)
                
            y.append(flag) 
       
        X_data_left = np.array(X_left)
        X_data_right = np.array(X_right)
        y_data = np.array(y)
        
        return X_data_left,X_data_right,y_data
            
        
        
    def load_data(self):
        # Read the the train set 
        # train-calibrated-shuffled
        current_path = os.getcwd()
        image_dir = os.path.join(current_path, "msl-images")
        image_files_directory = os.path.join(current_path, "msl-images\calibrated")
        
        train_dir = os.path.join(current_path, 'train')
       
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)
        #self.populate_data_for_comparator("train-calibrated-shuffled.txt",image_dir, train_dir,image_files_directory )
        
        validation_dir = os.path.join(current_path, 'validation')
        if not os.path.exists(validation_dir):
            os.mkdir(validation_dir)
        #self.copy_files("val-calibrated-shuffled.txt",image_dir, validation_dir,image_files_directory )
        
        print("Train directory--" + train_dir)
        # Create Images
        self.train_datagen = ImageDataGenerator(rescale=1./255)
        self.validation_generator = ImageDataGenerator(rescale=1./255)
        self.train_generator = self.train_datagen.flow_from_directory(
            train_dir,
            target_size=(255, 255),
            batch_size=20,
            class_mode='categorical')
        self.validation_generator = self.train_datagen.flow_from_directory(
            validation_dir,
            target_size=(255, 255),
            batch_size=20,
            class_mode='categorical')
        nb_train_samples = len(self.train_generator.filenames) 
        self.num_classes = len(self.train_generator.class_indices) 
        print(self.num_classes)
        self.X_train_data_left,self.X_train_data_right,self.y_train_data = self.populate_data_for_comparator("train-calibrated-shuffled.txt",image_dir, train_dir,image_files_directory )
        self.X_validate_data_left,self.X_validate_data_right,self.y_validate_data = self.populate_data_for_comparator("val-calibrated-shuffled.txt",image_dir, validation_dir,image_files_directory )

    def Create_Inception_V0(self):
        input_shape = (255, 255, 3)
        input_img = Input(shape = input_shape)
        start =  layers.Conv2D(64, (3, 3), activation='relu', input_shape=(255, 255, 3))
        root =    layers.MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_1_3x3/2')(start)
        tower_1 = layers.Conv2D(64, (1,1), padding='same', activation='relu')(root)
        tower_1 = layers.Conv2D(64, (3,3), padding='same', activation='relu')(tower_1)
        tower_2 = layers.Conv2D(64, (1,1), padding='same', activation='relu')(root)
        tower_2 = layers.Conv2D(64, (5,5), padding='same', activation='relu')(tower_2)
        tower_3 = layers.MaxPooling2D((3,3), strides=(1,1), padding='same')(root)
        tower_3 = layers.Conv2D(64, (1,1), padding='same', activation='relu')(tower_3)
        output = keras.layers.concatenate([tower_1, tower_2, tower_3], axis = 3)
        model = Model([input_img], output)
        return model

    def Build_Basic_Model(self):
        model = models.Sequential()
        model.add(layers.Conv2D(64, (3, 3), activation='relu',
        input_shape=(255, 255, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        return model
    def Compare(self):
        sequence_1_input  = Input(shape=(255,255,3), dtype='float32')
        sequence_2_input  = Input(shape=(255,255,3), dtype='float32')
        x_model = self.Build_Basic_Model()
        left_features = x_model(sequence_1_input)
        right_features = x_model(sequence_2_input)
        merged_features = layers.concatenate([left_features, right_features], axis=-1)
        predictions = layers.Dense(1, activation='sigmoid')(merged_features)
        model = Model([sequence_1_input, sequence_2_input], predictions)
        model.compile(loss='binary_crossentropy',
            optimizer=optimizers.RMSprop(lr=1e-4),
            metrics=['acc'])
        print(model.summary())
        callbacks = [
                keras.callbacks.TensorBoard(
                log_dir='E:\workdirectory\Code Name Val Halen\DS Sup\DL\Chapter 15\logs',
                histogram_freq=1
                )
                ]
        self.history = model.fit([self.X_train_data_left, self.X_train_data_right], self.y_train_data,
                                 validation_data=([self.X_validate_data_left, self.X_validate_data_right], self.y_validate_data),
                                 epochs=50, batch_size=10,
                                 callbacks=callbacks
                                 )
        
        
        
    def Execute(self):
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu',
        input_shape=(255, 255, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(self.num_classes, activation='softmax'))
        
        model.compile(loss='binary_crossentropy',
            optimizer=optimizers.RMSprop(lr=1e-4),
            metrics=['acc'])
        print(model.summary())
        self.history = model.fit_generator(
            self.train_generator,
            steps_per_epoch=100,
            epochs=30
            )
        
        
image_comparator = ImageComparator()
image_comparator.load_data()
image_comparator.Compare()