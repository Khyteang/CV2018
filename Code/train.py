import argparse
import functools
import itertools
import os
import six
import tensorflow as tf

import utils
from dataset_utils import NucleusDataset
from model import unet_v1, unet_argscope
from config import Configuration

from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops

tf.logging.set_verbosity(tf.logging.INFO)

def input_fn(split_name, is_training):
    """Create input graph for model.
    Args:
      split_name: one of 'train', 'validate' and 'eval'.
    Returns:
      two lists of tensors for features and labels, each of GPU_COUNT length.
    """
    with tf.device('/cpu:0'):
        dataset = NucleusDataset(split_name, is_training)
        feature_batch, mask_batch = dataset.load_batch()
        if Configuration.GPU_COUNT <= 1:
            # No GPU available or only 1 GPU.
            return [feature_batch], [mask_batch]

        # Note that passing num=batch_size is safe here, even though
        # dataset.batch(batch_size) can, in some cases, return fewer than batch_size
        # examples. This is because it does so only when repeating for a limited
        # number of epochs, but our dataset repeats forever.
        image_batch = tf.unstack(feature_batch[0], num=Configuration.BATCH_SIZE, axis=0)
        image_ids_batch = tf.unstack(feature_batch[1], num=Configuration.BATCH_SIZE, axis=0)
        
        feature_shards = [[] for i in range(Configuration.GPU_COUNT)]
        image_id_shards = [[] for i in range(Configuration.GPU_COUNT)]
                
        if split_name != "eval":
            mask_batch = tf.unstack(mask_batch, num=Configuration.BATCH_SIZE, axis=0)
            mask_shards = [[] for i in range(Configuration.GPU_COUNT)]
        else:
            mask_shards = None
        
        for i in range(Configuration.BATCH_SIZE):
            idx = i % Configuration.GPU_COUNT
            feature_shards[idx].append(image_batch[i])
            image_id_shards[idx].append(image_ids_batch[i])
            
            if split_name != "eval":
                mask_shards[idx].append(mask_batch[i])
                
        feature_shards = [(tf.parallel_stack(x), tf.parallel_stack(y))  for x, y in zip(feature_shards, image_id_shards)]
        
        if split_name != "eval":
            mask_shards = [tf.parallel_stack(x) for x in mask_shards]
            
        return feature_shards, mask_shards

def tower_fn(is_training, feature, mask):
    """Build computation tower
    Args:
        is_training: true if is training graph.
        feature: a Tensor.
        mask: a Tensor.
    Returns:
        A tuple with the loss for the tower, the gradients and parameters, and
        predictions.
    """
    image, image_id = feature
    with tf.contrib.framework.arg_scope(unet_argscope()):
        model_logits, end_points = unet_v1(image)
    
    '''
    now, logits is output with shape [batch_size x img h x img w x 1] 
    and represents probability of class 1
    '''
    logits = tf.reshape(model_logits, [-1])
    
    if mask is not None:
        masks = tf.reshape(mask, [-1])
        '''
        Eq. (1) The intersection part - tf.multiply is element-wise, 
        if logits were also binary then tf.reduce_sum would be like a bitcount here.
        '''
        inter = tf.reduce_sum(tf.multiply(logits, masks))

        '''
        Eq. (2) The union part - element-wise sum and multiplication, then vector sum
        '''
        union = tf.reduce_sum(tf.subtract(tf.add(logits, masks), tf.multiply(logits, masks)))

        # Eq. (4)
        tower_loss = tf.subtract(tf.constant(1.0, dtype = tf.float32), tf.div(inter, union))        
    else:
        tower_loss = None

    # For inference/visualization
    tower_pred = {
        'mask': model_logits,
        'probabilities': model_logits,
        'image_id': image_id
    }

    model_params = tf.trainable_variables()
    
    if mask is not None:
        tower_loss += Configuration.WEIGHT_DECAY * tf.add_n(
                [tf.nn.l2_loss(v) for v in model_params])

        g = tf.get_default_graph()
        # with g.gradient_override_map({'Relu': 'GuidedRelu'}):
        tower_grad = tf.gradients(tower_loss, model_params)    
        tower_grad = zip(tower_grad, model_params)
            
        tf.summary.image('features', image, family='unet1.3')
        tf.summary.image('orig_mask', tf.expand_dims(mask, 3), family='unet1.3')
        tf.summary.image('prediction', tf.expand_dims(model_logits, 3), family='unet1.3')
        
    else:
        tower_grad = None
        
    return tower_loss, tower_grad, tower_pred

def get_model_fn():
    """
    Returns a model function given the number of classes.
    """
    def model_fn(features, labels, mode):
        """This is the model body.
        Support single host, one or more GPU training. Parameter distribution can
        be either one of the following scheme.
        1. CPU is the parameter server and manages gradient updates.
        2. Parameters are distributed evenly across all GPUs, and the first GPU
        manages gradient updates.
        Args:
        features: a list of tensors, one for each tower
        labels: a list of tensors, one for each tower
        mode: ModeKeys.TRAIN or EVAL
        Returns:
        A EstimatorSpec object.
        """
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        
        tower_features = features
        tower_masks = labels
        tower_losses = []
        tower_gradvars = []
        tower_preds = []

        # channels first (NCHW) is normally optimal on GPU and channels last (NHWC)
        # on CPU. The exception is Intel MKL on CPU which is optimal with
        # channels_last.
        data_format = None
        if not data_format:
            if Configuration.GPU_COUNT == 0:
                data_format = 'channels_last'
            else:
                data_format = 'channels_first'

        if Configuration.GPU_COUNT == 0:
            num_devices = 1
            device_type = 'cpu'
        else:
            num_devices = Configuration.GPU_COUNT
            device_type = 'gpu'
 
        for i in range(num_devices):
            worker_device = '/{}:{}'.format(device_type, i)
            if Configuration.VARIABLE_STRATEGY == 'CPU':
                device_setter = utils.local_device_setter(
                    worker_device=worker_device)
            elif Configuration.VARIABLE_STRATEGY == 'GPU':
                device_setter = utils.local_device_setter(
                    ps_device_type='gpu',
                    worker_device=worker_device,
                    ps_strategy=tf.contrib.training.GreedyLoadBalancingStrategy(
                        Configuration.GPU_COUNT, tf.contrib.training.byte_size_load_fn))
            with tf.variable_scope('', reuse=bool(i != 0)):
                with tf.name_scope('tower_%d' % i) as name_scope:
                    with tf.device(device_setter):
                        loss, gradvars, preds = tower_fn(
                            is_training, tower_features[i], tower_masks and tower_masks[i])
                        tower_losses.append(loss)
                        tower_gradvars.append(gradvars)
                        tower_preds.append(preds)
                        if i == 0:
                            # Only trigger batch_norm moving mean and variance update from
                            # the 1st tower. Ideally, we should grab the updates from all
                            # towers but these stats accumulate extremely fast so we can
                            # ignore the other stats from the other towers without
                            # significant detriment.
                            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,
                                                           name_scope)
        if mode == 'train' or mode == 'eval':
            # Now compute global loss and gradients.
            gradvars = []
            with tf.name_scope('gradient_ing'):
                all_grads = {}
                for grad, var in itertools.chain(*tower_gradvars):
                    if grad is not None:
                        all_grads.setdefault(var, []).append(grad)
                for var, grads in six.iteritems(all_grads):
                    # Average gradients on the same device as the variables
                    # to which they apply.
                    with tf.device(var.device):
                        if len(grads) == 1:
                            avg_grad = grads[0]
                        else:
                            avg_grad = tf.multiply(
                                tf.add_n(grads), 1. / len(grads))
                    gradvars.append((avg_grad, var))

            # Device that runs the ops to apply global gradient updates.
            consolidation_device = '/gpu:0' if Configuration.VARIABLE_STRATEGY == 'GPU' else '/cpu:0'
            with tf.device(consolidation_device):
                loss = tf.reduce_mean(tower_losses, name='loss')

                examples_sec_hook = utils.ExamplesPerSecondHook(
                    Configuration.BATCH_SIZE, every_n_steps=10)

                global_step = tf.train.get_global_step()

                learning_rate = tf.constant(Configuration.LEARNING_RATE)

                tensors_to_log = {'learning_rate': learning_rate, 'training_iou': loss}

                logging_hook = tf.train.LoggingTensorHook(
                    tensors=tensors_to_log, every_n_iter=100)
 
                initializer_hook = utils.IteratorInitializerHook()

                train_hooks = [initializer_hook,
                               logging_hook, examples_sec_hook]

                optimizer = tf.train.MomentumOptimizer(
                    learning_rate=Configuration.LEARNING_RATE, momentum=Configuration.MOMENTUM)

                # Create single grouped train op
                train_op = [
                    optimizer.apply_gradients(
                        gradvars, global_step=global_step)
                ]
                train_op.extend(update_ops)
                train_op = tf.group(*train_op)

                predictions = {
                    'mask':
                        tf.concat([p['mask'] for p in tower_preds], axis=0),
                    'probabilities':
                        tf.concat([p['probabilities']
                                   for p in tower_preds], axis=0),
                    'image_id':
                        tf.concat([p['image_id']
                                   for p in tower_preds], axis=0)
                }
                stacked_masks = tf.concat(labels, axis=0)
                
                tf.summary.image('prediction', tf.concat([tf.expand_dims(p['mask'], 3) for p in tower_preds], axis=0), family='mean_iou')
                tf.summary.image('mask', tf.expand_dims(stacked_masks, 3), family='mean_iou')
        
                metrics = {
                    'mean_iou':
                        tf.metrics.mean_iou(
                            labels=stacked_masks, predictions=predictions['mask'], num_classes=2)
                }


            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=loss,
                train_op=train_op,
                training_hooks=train_hooks,
                eval_metric_ops=metrics)
        else:
            predictions = {
                'mask':
                    tf.concat([p['mask'] for p in tower_preds], axis=0),
                'probabilities':
                    tf.concat([p['probabilities']
                               for p in tower_preds], axis=0),
                'image_id':
                        tf.concat([p['image_id']
                                   for p in tower_preds], axis=0),
                'features': tf.concat([feature[0] for feature in features], axis=0)
            }
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions)
    return model_fn

def get_experiment_fn():
    """
    Returns the experiment function given a dataset directory.
    The dataset directory needs train.tfrecord and validate.tfrecord present.
    """
    def experiment_fn(run_config, hparams):
        """
        This is a method passed to tf.contrib.learn.learn_runner that will
        return an instance of an Experiment.
        """

        train_input_fn = functools.partial(
            input_fn,
            split_name='train',
            is_training=True)

        eval_input_fn = functools.partial(
            input_fn,
            split_name='validation',
            is_training=False)

        classifier = tf.estimator.Estimator(
            model_fn=get_model_fn(),
            config=run_config)

        return tf.contrib.learn.Experiment(
            classifier,
            train_input_fn=train_input_fn,
            eval_input_fn=eval_input_fn,
            train_steps=None,  # Train forever
            eval_steps=Configuration.VALIDATION_STEPS)
    return experiment_fn


def train():
    """
    Creates the session configuration and starts the TensorFlow LearnRunner.
    This will begin training using the configuration specified in config.Configuration
    """
    # Session configuration.
    sess_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        intra_op_parallelism_threads=0,  # Autocompute how many threads to run
        gpu_options=tf.GPUOptions(force_gpu_compatible=True))

    config = tf.contrib.learn.RunConfig(
        session_config=sess_config, model_dir=Configuration.MODEL_SAVE_DIR)
    tf.contrib.learn.learn_runner.run(
        get_experiment_fn(),
        run_config=config,
        hparams=tf.contrib.training.HParams())


if __name__ == '__main__':
    # A (supposed) 5% percent boost in certain GPUs by using faster convolution operations
    os.environ['TF_SYNC_ON_FINISH'] = '0'
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    train()
