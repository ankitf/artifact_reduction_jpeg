
from ds_loader import DSLoader
from keras.models import Sequential
from keras.layers import Input, Conv2D, Activation
from keras.optimizers import SGD
import os


class ARCNN:
    ''' ARCNN model for jpeg artifact reduction. '''
    def __init__(self, dataset_path='./dataset/', learning_rate=0.0005, log_path='./logs', jpeg_quality=0.6, block_size=32, block_channels=3, train_batch_size=128,
                 train_stride=10, validation_batch_size=256, validation_stride=32):
        self.log_path = log_path
        self.learning_rate = learning_rate
        self.jpeg_quality = jpeg_quality
        self.train_batch_size = train_batch_size
        self.validation_batch_size = validation_batch_size
        self.block_size = block_size
        self.train_stride = train_stride
        self.validation_stride = validation_stride
        self.model = []
        self.channels = block_channels
        self.dataset_path = dataset_path
        self.train_dataset_path = os.path.join(self.dataset_path, 'train')
        self.validation_dataset_path = os.path.join(self.dataset_path, 'validation')
        self.train_dataset = DSLoader(self.train_dataset_path, self.jpeg_quality, self.block_size,
                                      self.channels, self.train_stride)
        self.validation_dataset = DSLoader(self.validation_dataset_path, self.jpeg_quality, self.block_size,
                                           self.channels, self.validation_stride)

    def build_model(self, learning_rate):
        self.model = Sequential()
        self.model.add(Conv2D(64, (9, 9), input_shape=(self.block_size, self.block_size, self.channels), kernel_initializer='random_normal'))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(32, (7, 7), kernel_initializer='random_normal'))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(16, (1, 1), kernel_initializer='random_normal'))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(self.channels, (5, 5), kernel_initializer='random_normal'))

        optimizer = SGD(lr=self.learning_rate)
        self.model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['binary_accuracy'])
        return

    def train_model(self, number_of_train_iterations):
        
        return
     
        
        
