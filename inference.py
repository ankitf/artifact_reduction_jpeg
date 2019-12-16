
import os
import config
import math
import cv2
import numpy as np
from keras.models import load_model
from skimage.measure import compare_ssim
import pdb


def main():
    output_dir = './results'
    inference_dir = './dataset/validation/'

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        
    images_paths = []
    for path, folders, files in os.walk(inference_dir):
        for filename in files:
            if filename.endswith('.bmp') or filename.endswith('.png'):
                filepath = os.path.join(path, filename)
                images_paths.append(filepath)
    
    # inference_image_path = './results/flowers.png'
    ssim = np.zeros([len(images_paths)], np.float32)
    
    for c, inference_image_path in enumerate(images_paths):
        base_name = os.path.splitext(os.path.basename(inference_image_path))[0]
        save_original_image_path = os.path.join(output_dir, base_name + '.png')
        jpeg_image_path = base_name + '_jpeg' + str(int(config.jpeg_quality*100)) + '.jpeg'
        jpeg_image_path =  os.path.join(output_dir, jpeg_image_path)
        reconstructed_image_path = base_name + '_reconstructed.jpeg'
        reconstructed_image_path =  os.path.join(output_dir, reconstructed_image_path)

        original_image = cv2.imread(inference_image_path)
        
        cv2.imwrite(save_original_image_path, original_image)
        cv2.imwrite(jpeg_image_path, original_image, [int(cv2.IMWRITE_JPEG_QUALITY), config.jpeg_quality])

        image_height, image_width = original_image.shape[:2]

        target_width = math.ceil(image_width / config.block_size) * config.block_size
        target_height = math.ceil(image_height / config.block_size) * config.block_size

        resized_image = cv2.resize(original_image, (target_width, target_height))

        cols = target_width // config.block_size
        rows = target_height // config.block_size


        # making image patches
        print('Making image patches')
        total_patches = rows * cols
        X = np.zeros([total_patches, config.block_size, config.block_size, config.block_channels], np.float32)
        count = 0
        for i in range(rows):
            for j in range(cols):
                start_col = j * config.block_size
                end_col = start_col + config.block_size
                start_raw = i * config.block_size
                end_raw = start_raw + config.block_size
                # X[count, :, :, :] = resized_image[start_raw: end_raw, start_col: end_col, :] / 255.0
                X[count, :, :, :] = resized_image[start_raw: end_raw, start_col: end_col, :] / 255.0
                count += 1

        print('Running inferece... ')
        model = load_model('model.h5')
        y_pred = model.predict_on_batch(X)

        # make image from patches
        reconstructed_image = np.zeros([target_height, target_width, config.block_channels])
        count= 0
        for i in range(rows):
            for j in range(cols):
                start_col = j * config.block_size
                end_col = start_col + config.block_size
                start_raw = i * config.block_size
                end_raw = start_raw + config.block_size
                reconstructed_image[start_raw:end_raw, start_col:end_col, :] = X[count]
                count += 1

        reconstructed_image = reconstructed_image * 255        
        reconstructed_image =  cv2.resize(reconstructed_image, (image_width, image_height))
        print('Saving the reconstructed_image...')
        cv2.imwrite(reconstructed_image_path, reconstructed_image)

        ssim[c] = compare_ssim(original_image, reconstructed_image, multichannel=True)


    print('Mean SSIM for given images: {}'.format(ssim.mean()))


if __name__ == '__main__':
    main()
