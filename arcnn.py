
from ds_loader import DSLoader
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, Conv2D, Activation
from keras.optimizers import SGD
import math
import os



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

    def train_model(self, number_of_train_iterations=50):
        print('[INFO]loading the training dataset...')
        self.train_dataset.load_dataset()
        print('[INFO]loading the validation dataset...')
        self.validation_dataset.load_dataset()

        train_losses = []
        validations_losses = []

        train_psnr = []
        validation = psnr[]
        def calculate_psnr(mse):
            return 10.0 * math.log10(1.0/mse)
        print('[INFO]training the model...')
        
        for i in range(number_of_train_iterations):
            trainX, trainY = self.train_dataset.get_batch()
            train_loss = self.model.train_on_batch(trainX, trainY)
            
            psnr = calculate_psnr(train_loss)
            print('MSE: {}, PSNR: {}'.format(train_loss, psnr))
            

            
        return
     
        
        
