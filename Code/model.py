from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

slim = tf.contrib.slim

def unet_v1(inputs, scope = None, final_endpoint = "mask"):
    end_points = {}

    def add_and_check_final(name, net):
        end_points[name] = net
        print("Added %s of shape: %s" % (name, net.shape))
        return name == final_endpoint
    
    def conv_pool(name, inputs, num_filter, filter_size = 3, max_pool_size = 2, padding = 'VALID'):
        net = slim.conv2d(inputs, num_filter, filter_size, activation_fn=tf.nn.relu, padding = padding)
        net = slim.conv2d(net, num_filter, filter_size, activation_fn=tf.nn.relu, padding = padding)
        if add_and_check_final(name, net): return net, end_points

        pool = slim.max_pool2d(net, max_pool_size, stride=2, padding=padding)
        return pool

    def crop_deconv_merge(name, inputs, num_filter, output_num_filter, filter_size = 3, padding = 'VALID'):
        batch_size = inputs[1].shape[0]
        width = inputs[1].shape[1]
        height = inputs[1].shape[2]
        depth = inputs[1].shape[3]

        net_up_sample = tf.image.resize_nearest_neighbor(inputs[0], (width * 2, height * 2))
        import pdb; pdb.set_trace()
#         slim.conv2d_transpose(inputs[0], output/_num_filter, kernel_size = filter_size, stride=2, activation_fn=tf.nn.relu)
        
        width_start_idx = int((width - net_up_sample.shape[1]).value / 2)
        height_start_idx = int((height - net_up_sample.shape[2]).value / 2)
        
        crop = tf.slice(inputs[1], [0, width_start_idx, height_start_idx, 0], [batch_size, net_up_sample.shape[1].value, net_up_sample.shape[2].value, depth])
        
        merge_net = tf.concat([net_up_sample, crop], 3)
        net = slim.conv2d(merge_net, num_filter, filter_size, activation_fn=tf.nn.relu, padding = padding )
        net = slim.conv2d(net, num_filter, filter_size, activation_fn=tf.nn.relu, padding = padding)
        return net


    with tf.variable_scope(scope, 'UnetV1', [inputs]):
        pool1 = conv_pool('net1', inputs, 64)
        pool2 = conv_pool('net2', pool1, 128)
        pool3 = conv_pool('net3', pool2, 256)
        pool4 = conv_pool('net4', pool3, 512)

        net5 = slim.conv2d(pool4, 1024, 3, activation_fn=tf.nn.relu, padding = 'VALID')
        net5 = slim.conv2d(net5, 1024, 3, activation_fn=tf.nn.relu, padding = 'VALID')
        if add_and_check_final('net5', net5): return net5, end_points

        net6 = crop_deconv_merge('net6', (net5, end_points['net4']), 512, 512, )
        net7 = crop_deconv_merge('net7', (net6, end_points['net3']), 256, 256)
        net8 = crop_deconv_merge('net8', (net7, end_points['net2']), 128, 128)
        net9 = crop_deconv_merge('net9', (net8, end_points['net1']), 64, 64)
        if add_and_check_final('net9', net9): return net9, end_points
        
        premask = slim.conv2d(net9, 1, 1, activation_fn=tf.nn.sigmoid, padding = 'VALID')
        
        # Remove the "channel" dimension from the mask since its just 1 axis (1 channel)
        mask = tf.squeeze(premask)
        
        if add_and_check_final('mask', mask): return mask, end_points
        
        return mask, end_points

def unet_argscope(weight_decay=0.00004,
                        use_batch_norm=True,
                        batch_norm_decay=0.9997,
                        batch_norm_epsilon=0.001,
                        activation_fn=tf.nn.relu):
  """Defines the default arg scope for UNet models.

  Args:
    weight_decay: The weight decay to use for regularizing the model.
    use_batch_norm: "If `True`, batch_norm is applied after each convolution.
    batch_norm_decay: Decay for batch norm moving average.
    batch_norm_epsilon: Small float added to variance to avoid dividing by zero
      in batch norm.
    activation_fn: Activation function for conv2d.

  Returns:
    An `arg_scope` to use for the inception models.
  """
  batch_norm_params = {
      # Decay for the moving averages.
      'decay': batch_norm_decay,
      # epsilon to prevent 0s in variance.
      'epsilon': batch_norm_epsilon,
      # collection containing update_ops.
      'updates_collections': tf.GraphKeys.UPDATE_OPS,
      # use fused batch norm if possible.
      'fused': None,
  }
  if use_batch_norm:
    normalizer_fn = slim.batch_norm
    normalizer_params = batch_norm_params
  else:
    normalizer_fn = None
    normalizer_params = {}
  # Set weight_decay for weights in Conv and FC layers.
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      weights_regularizer=slim.l2_regularizer(weight_decay)):
    with slim.arg_scope(
        [slim.conv2d],
        weights_initializer=slim.variance_scaling_initializer(),
        activation_fn=activation_fn,
        normalizer_fn=normalizer_fn,
        normalizer_params=normalizer_params) as sc:
      return sc