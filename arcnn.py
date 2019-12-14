
from ds_loader import DSLoader
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, Conv2D, Activation
from keras.optimizers import SGD
import numpy as np
import math
import os
import shutil



class ARCNN:
    ''' ARCNN model for jpeg artifact reduction. '''
    def __init__(self, dataset_path='./dataset/', log_path='./logs', jpeg_quality=0.6, block_size=32,
                 block_channels=3, train_batch_size=128, train_stride=10, validation_batch_size=256, validation_stride=32):
        self.log_path = log_path
        self.jpeg_quality = jpeg_quality
        self.train_batch_size = train_batch_size
        self.validation_batch_size = validation_batch_size
        self.block_size = block_size
        self.train_stride = train_stride
        self.validation_stride = validation_stride
        self.channels = block_channels
        self.dataset_path = dataset_path
        self.train_dataset_path = os.path.join(self.dataset_path, 'train')
        self.validation_dataset_path = os.path.join(self.dataset_path, 'validation')
        self.train_dataset = DSLoader(self.train_dataset_path, self.jpeg_quality, self.block_size,
                                      self.channels, self.train_batch_size, self.train_stride)
        self.validation_dataset = DSLoader(self.validation_dataset_path, self.jpeg_quality, self.block_size,
                                           self.channels, self.validation_batch_size, self.validation_stride)
        self.model = []
        # removing old logs 
        if os.path.exists(self.log_path):
            shutil.rmtree(self.log_path)
        self.summary_writer = tf.summary.FileWriter(self.log_path)

    def build_model(self, learning_rate):
        self.learning_rate = learning_rate
        self.model = Sequential()
        self.model.add(Conv2D(64, (9, 9), input_shape=(self.block_size, self.block_size, self.channels),
                              padding='same', kernel_initializer='random_normal'))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(32, (7, 7),padding='same', kernel_initializer='random_normal'))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(16, (1, 1), padding='same', kernel_initializer='random_normal'))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(self.channels, (5, 5), padding='same', kernel_initializer='random_normal'))
        optimizer = SGD(lr=self.learning_rate)

        self.model.compile(loss='mean_squared_error', optimizer=optimizer)
        return

    def _write_logs_to_tensorboard(self, current_iteration, train_psnr, validation_psnr):
        # train psnr
        summary = tf.Summary()
        
        value = summary.value.add()
        value.simple_value = train_psnr
        value.tag = 'Train PSNR'
        
        value = summary.value.add()
        value.simple_value = validation_psnr
        value.tag = 'Validation PSNR'
        
        self.summary_writer.add_summary(summary, current_iteration)
        self.summary_writer.flush()
                
    def train_model(self, number_of_train_iterations=50):
        print('[INFO]loading the training dataset...')
        self.train_dataset.load_dataset()
        print('[INFO]loading the validation dataset...')
        self.validation_dataset.load_dataset()

        train_losses = []
        validations_losses = []

        train_psnr = []
        validation = []

        def mean_squared_error(y_true, y_pred):
            mse = np.mean((y_true - y_pred)**2)
            return mse

        def calculate_psnr(mse):
            return 10.0 * math.log10(1.0/mse)

        print('[INFO]training the model...')
        
        for i in range(number_of_train_iterations):
            trainX, trainY = self.train_dataset.get_batch()
            train_mse = self.model.train_on_batch(trainX, trainY)
            train_psnr = calculate_psnr(train_mse)
            
            validationX, validationY = self.validation_dataset.get_batch()
            y_pred = self.model.predict_on_batch(validationX)
            validation_mse = mean_squared_error(validationY, y_pred)
            validation_psnr = calculate_psnr(validation_mse)
    
            self._write_logs_to_tensorboard(i, train_psnr, validation_psnr)
            print('Train MSE: {:.4f}, Train PSNR: {:.4f}, Validation MSE: {:.4f}, Validation PSNR: {:.4f}'.
                  format(train_mse, train_psnr, validation_mse, validation_psnr))
                        
        return
     
        
        
