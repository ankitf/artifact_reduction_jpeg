
import tensorflow as tf
from keras.preprocessing.image import load_img, array_to_img, img_to_array
import numpy as np
import os
import cv2



class DSLoader:
    ''' Dataset loader for JPEG artifact reduction. '''
    def __init__(self, dataset_path, quality, block_size, block_channel, batch_size, stride):
        self.dataset_path = dataset_path
        self.images_paths = self._load_paths(self.dataset_path)
        self.jpeg_quality = quality
        self.original_data = []
        self.compressed_data = []
        self.block_size = block_size
        self.block_channels = block_channel 
        self.batch_size = 128
        self.current_index = 0
        self.stride = stride
        
    def _load_paths(self, dataset_path):
        images_paths = []
        for path, folders, files in os.walk(dataset_path):
            for filename in files:
                if filename.endswith('.bmp') or filename.endswith('.png'):
                    filepath = os.path.join(path, filename)
                    images_paths.append(filepath)
        return images_paths
    
    def _get_jpeg_encoded_image(self, original_image_path, quality):
        original_image = cv2.imread(original_image_path)
        cv2.imwrite('tmp.jpeg', original_image, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        compressed_image = cv2.imread('tmp.jpeg')
        os.remove('tmp.jpeg')
        return original_image, compressed_image

    def get_batch(self):
        X = np.zeros([self.batch_size, self.block_size, self.block_size, self.channels], np.float32)
        Y = np.zeros([self.batch_size, self.block_size, self.block_size, self.channels], np.float32)
        if self.current_index > len(self.compressed_data) - self.batch_size:
            self.current_index = 0
            return None, None

        for i in range(self.batch_size):
            X[i, :, :, :] = self.compressed_data[self.current_index+i] / 255.0
            Y[i, :, :, :] = self.original_data[self.current_index+i] / 255.0

        self.current_index += self.batch_size
        return X, Y
        
    
    def load_dataset(self):
        for image_file in self.images_paths:
            original_image, compressed_image = self._get_jpeg_encoded_image(image_file, self.jpeg_quality)
            # create image blocks and add to data
            rows = (original_image.shape[0] - self.block_size) // self.stride + 1
            cols = (original_image.shape[1] - self.block_size) // self.stride + 1
            for row in range(rows):
                for col in range(cols):
                    start_col = col * self.stride
                    start_row = row * self.stride
                    original_block = original_image[start_row: start_row+self.block_size,
                                           start_col:start_col+self.block_size, :]
                    compressed_block = compressed_image[start_row: start_row+self.block_size,
                                           start_col:start_col+self.block_size, :]
                    self.original_data.append(original_block)
                    self.compressed_data.append(compressed_block)
        return 
        
        
    
train_dataset_path = './dataset/train/'
block_size = 32
quality = 0.6
block_channel = 3
batch_size = 128
train_image_stride = 10

validation_dataset_path = './dataset/validation'

train_data = DSLoader(train_dataset_path, quality, block_size, block_channel, batch_size, train_image_stride)
# validation_data = DSLoader(validation_dataset_path)
import pdb
train_data.load_dataset()
