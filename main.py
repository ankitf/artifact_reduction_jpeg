
from arcnn import ARCNN
import config

def main():
    
    arcnn = ARCNN(config.dataset_path, config.log_path, config.jpeg_quality, config.block_size,
                  config.block_channels, config.train_batch_size, config.train_stride,
                  config.validation_batch_size, config.validation_stride)
    arcnn.build_model(config.learning_rate)
    arcnn.train_model(config.number_of_train_iterations)
    return

if __name__=='__main__':
    main()
