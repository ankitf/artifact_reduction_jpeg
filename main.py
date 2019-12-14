
from arcnn import ARCNN

def main():
    dataset_path = './dataset/'
    log_path = './logs'
    
    train_batch_size = 128
    validation_batch_size = 256
    block_size = 32
    block_channels = 3
    train_stride = 10
    validation_stride = 32
    jpeg_quality = 0.6
    
    number_of_train_iterations = 50
    learning_rate = 0.0006

    arcnn = ARCNN(dataset_path, log_path, jpeg_quality, block_size, block_channels, train_batch_size,
                  train_stride, validation_batch_size, validation_stride)
    arcnn.build_model(learning_rate)
    arcnn.train_model(number_of_train_iterations)
    return

if __name__=='__main__':
    main()
