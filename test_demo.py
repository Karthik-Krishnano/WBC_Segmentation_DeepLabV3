from os import path as osp

import numpy as np

from model import DeepLab
from tqdm import trange
from utils import (Dataset, Iterator, save_load_means, subtract_channel_means,
                   validation_single_demo)

if __name__ == '__main__':

    data_dir = 'E:/PRO/WBC_Segment/DeepLab-V3/data/custom_dataset/'#'data/datasets/VOCdevkit/VOC2012/'
    testset_filename = osp.join(data_dir, 'dataset1.txt')
    images_dir = osp.join(data_dir, 'Images/')
    labels_dir = osp.join(data_dir, 'Labels/')
    demo_dir = 'E:/PRO/WBC_Segment/DeepLab-V3/data/custom_dataset/outputs/'#'data/demos/deeplab/resnet_101_voc2012/'
    models_dir = 'E:/PRO/WBC_Segment/DeepLab-V3/data/models/custom_wbc/resnet_101_voc2012'#'data/models/deeplab/resnet_101_voc2012/'
    model_filename = 'resnet_101_0.4011.ckpt'

    channel_means = save_load_means(means_filename='channel_means.npz', image_filenames=None)

    minibatch_size = 16

    test_dataset = Dataset(dataset_filename=testset_filename, images_dir=images_dir, labels_dir=labels_dir, image_extension='.bmp', label_extension='.png')
    test_iterator = Iterator(dataset=test_dataset, minibatch_size=minibatch_size, process_func=None, random_seed=None, scramble=False, num_jobs=1)

    deeplab = DeepLab('resnet_101', training=False)
    deeplab.load(osp.join(models_dir, model_filename))

    n_samples = 8
    for i in trange(n_samples):
        image, label = test_iterator.next_raw_data()
        image_input = subtract_channel_means(image=image, channel_means=channel_means)

        output = deeplab.test(inputs=[image_input], target_height=image.shape[0], target_width=image.shape[1])[0]

        validation_single_demo(image, np.squeeze(label, axis=-1), np.argmax(output, axis=-1), demo_dir, str(i))

    deeplab.close()
