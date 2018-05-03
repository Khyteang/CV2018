import tensorflow as tf
import os

from tensorflow.python.ops.parsing_ops import parse_single_example
from tensorflow.python.ops import array_ops
from config import Configuration
from tensorflow.contrib.slim.python.slim.data.dataset import Dataset
from tensorflow.contrib.slim.python.slim.data.data_decoder import DataDecoder
from tensorflow.contrib.slim.python.slim.data.dataset_data_provider import DatasetDataProvider

import image_preprocessing

class NucleusDataDecoder(DataDecoder):
    def __init__(self, has_mask):
        if has_mask:
            self.keys_to_features = {
                'mask': tf.FixedLenFeature((), dtype=tf.string),
                'image': tf.FixedLenFeature((), dtype=tf.string),
                'image_id': tf.FixedLenFeature((), dtype=tf.string),
                'width': tf.FixedLenFeature([], tf.int64),
                'height': tf.FixedLenFeature([], tf.int64)
            }

            self.items_to_descriptions = {
                'mask': "The image mask",
                'image': "The original image",
                'image_id': "The original image id",
                'width': "Width of original image",
                'height': "Height of original image"
            }
        else:
            self.keys_to_features = {
                'image': tf.FixedLenFeature((), dtype=tf.string),
                'image_id': tf.FixedLenFeature((), dtype=tf.string),
                'width': tf.FixedLenFeature([], tf.int64),
                'height': tf.FixedLenFeature([], tf.int64)
            }

            self.items_to_descriptions = {
                'image': "The original image",
                'image_id': "The original image id",
                'width': "Width of original image",
                'height': "Height of original image"
            }
            

    def decode(self, data, items): 
        """Decodes the data to returns the tensors specified by the list of items.
        Args:
        data: A possibly encoded data format.
        items: A list of strings, each of which indicate a particular data type.
        Returns:
        A list of `Tensors`, whose length matches the length of `items`, where
        each `Tensor` corresponds to each item.
        Raises:
        ValueError: If any of the items cannot be satisfied.
        """
        example = parse_single_example(data, self.keys_to_features)

        return [example[i] for i in items]
        

    def list_items(self):
        """Lists the names of the items that the decoder can decode.
        Returns:
        A list of string names.
        """
        return list(self.keys_to_features.keys())

class NucleusDataset():
    def __init__(self, split_name, is_training=True):
        if split_name not in ['train', 'validation', 'eval']:
            raise ValueError(
                'The split_name %s is not recognized. Please input either \
                train or validation as the split_name' % (split_name))
        
        self.reader = None
        self.decoder = None
        self.is_training = is_training
        self.is_eval = split_name == 'eval'
        self.split_name = split_name
        self.dataset = self._get_split(split_name)

    def load_batch(self):
        '''
        Loads a batch for training.
        OUTPUTS:
        - images(Tensor): a Tensor of the shape (batch_size, height, width, channels) that contain one batch of images
        - masks(Tensor): the batch's masks with the shape (batch_size,).
        '''
        # First create the data_provider object
        data_provider = DatasetDataProvider(
            self.dataset,
            common_queue_capacity = 8 * Configuration.BATCH_SIZE,
            common_queue_min = 3 * Configuration.BATCH_SIZE,
            shuffle=self.split_name != "eval")

        items = ['image', 'image_id', 'width', 'height']
        if self.split_name != "eval":
            items.append('mask')
            image, image_id, width, height, mask = data_provider.get(items)
        else:
            image, image_id, width, height = data_provider.get(items)
            mask = None
            
        height = tf.cast(height, tf.int32)
        width = tf.cast(width, tf.int32)
        image = tf.decode_raw(image, tf.float32)
        image = tf.reshape(image, (width, height, 3))
        
        if self.split_name != "eval":
            mask = tf.decode_raw(mask, tf.float32)
            mask = tf.reshape(mask, (width, height, 2))
            mask = mask[:,:,1]
            mask = tf.greater(mask, 0)
            mask = tf.cast(mask, tf.float32)
        
        # Perform the correct preprocessing for this image depending if it is training or evaluating
        processed_image, processed_mask = image_preprocessing.preprocess_image(image, mask, Configuration.SCALED_IMAGE_SIZE, Configuration.SCALED_MASK_SIZE, self.is_eval)
        
        
        visualize_image = tf.image.resize_bilinear(tf.expand_dims(image, 0), [Configuration.SCALED_IMAGE_SIZE, Configuration.SCALED_IMAGE_SIZE],
                                      align_corners=False)
        visualize_image = tf.squeeze(visualize_image, 0)
        
        # Batch up the image by enqueing the tensors internally in a FIFO queue and dequeueing many elements with tf.train.batch.
        
        if self.is_eval:
            images, raw, image_ids = tf.train.batch(
                [processed_image, visualize_image, image_id],
                batch_size = Configuration.BATCH_SIZE,
            num_threads = 8,
            capacity = 4 * Configuration.BATCH_SIZE,
            allow_smaller_final_batch = True)
            
            tf.summary.image('uncropped_image', raw, family='unet1.3')
            
            return (images, image_ids), None
        else:
            images, raw, image_ids, masks = tf.train.batch(
                [processed_image, visualize_image, image_id, processed_mask],
                batch_size = Configuration.BATCH_SIZE,
            num_threads = 8,
            capacity = 4 * Configuration.BATCH_SIZE,
            allow_smaller_final_batch = True)
            
            tf.summary.image('uncropped_image', raw, family='unet1.3')
            
            return (images, image_ids), masks

    
    def _get_split(self, split_name):
        '''
        Obtains the split - training or validation - to create a Dataset class for feeding the examples into a queue later on. This function will
        set up the decoder and dataset information all into one Dataset class so that you can avoid the brute work later on.
        INPUTS:
        - split_name(str): 'train' or 'validation'. Used to get the correct data split of tfrecord files
        OUTPUTS:
        - dataset (Dataset): A Dataset class object where we can read its various components for easier batch creation later.
        '''

        # First check whether the split_name is train or validation
        
        dataset_dir = Configuration.DATASET_DIR if split_name != "eval" else Configuration.EVAL_DATASET_DIR
        tfrecord_path = os.path.join("%s" % dataset_dir, "%s.tfrecord" % split_name)

        # Count the total number of examples in all of these shard
        num_samples = sum(1 for i in tf.python_io.tf_record_iterator(tfrecord_path))

        # Create a reader, which must be a TFRecord reader in this case
        self.reader = tf.TFRecordReader

        # Start to create the decoder
        self.decoder = NucleusDataDecoder(split_name != "eval")

        # Actually create the dataset
        dataset = Dataset(
            data_sources = tfrecord_path,
            decoder = self.decoder,
            reader = self.reader,
            num_readers = 8,
            num_samples = num_samples,
            items_to_descriptions = self.decoder.items_to_descriptions)

        return dataset