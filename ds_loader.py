
import tensorflow as tf
import os


class DSLoader:
    ''' Dataset loader for JPEG artifact reduction. '''
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.train_path = os.path.join(self.dataset_path, 'train')
        self.validation_path = os.path.join(self.dataset_path, 'validation')
        self.train_images_paths = self._load_dataset(self.train_path)
        self.validation_images_paths = self._load_dataset(self.validation_path)
        
    def _load_dataset(self, dataset_path):
        images_paths = []
        for path, folders, files in os.walk(dataset_path):
            for filename in files:
                if filename.endswith('.bmp') or filename.endswith('.png'):
                    filepath = os.path.join(path, filename)
                    images_paths.append(filepath)
        return images_paths


dataset_path = './dataset/'
dsloader = DSLoader(dataset_path)
import pdb
pdb.set_trace()
