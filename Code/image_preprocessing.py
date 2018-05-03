#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides utilities to preprocess images for the Inception networks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import tensorflow as tf

from tensorflow.python.ops import control_flow_ops


def apply_with_random_selector(x, func, num_cases):
  """Computes func(x, sel), with sel sampled from [0...num_cases-1].

  Args:
    x: input Tensor.
    func: Python function to apply.
    num_cases: Python int32, number of cases to sample sel from.

  Returns:
    The result of func(x, sel), where func receives the value of the
    selector as a python integer, but sel is sampled dynamically.
  """
  sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
  # Pass the real x only to one of the func calls.
  return control_flow_ops.merge([
      func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
      for case in range(num_cases)])[0]


def distort_color(image, color_ordering=0, fast_mode=True, scope=None):
  """Distort the color of a Tensor image.

  Each color distortion is non-commutative and thus ordering of the color ops
  matters. Ideally we would randomly permute the ordering of the color ops.
  Rather then adding that level of complication, we select a distinct ordering
  of color ops for each preprocessing thread.

  Args:
    image: 3-D Tensor containing single image in [0, 1].
    color_ordering: Python int, a type of distortion (valid values: 0-3).
    fast_mode: Avoids slower ops (random_hue and random_contrast)
    scope: Optional scope for name_scope.
  Returns:
    3-D Tensor color-distorted image on range [0, 1]
  Raises:
    ValueError: if color_ordering not in [0, 3]
  """
  with tf.name_scope(scope, 'distort_color', [image]):
    if fast_mode:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      else:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    else:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
      elif color_ordering == 2:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      elif color_ordering == 3:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
      else:
        raise ValueError('color_ordering must be in [0, 3]')

    # The random_* ops do not necessarily clamp.
    return tf.clip_by_value(image, 0.0, 1.0)


def distorted_bounding_box_crop(image,
                                mask,
                                image_size,
                                mask_size,
                                min_object_covered=0.1,
                                aspect_ratio_range=(0.75, 1.33),
                                area_range=(0.05, 1.0),
                                max_attempts=100,
                                scope=None):
  """Generates cropped_image using a one of the bboxes randomly distorted.

  See `tf.image.sample_distorted_bounding_box` for more documentation.

  Args:
    image: 3-D Tensor of image (it will be converted to floats in [0, 1]).
    min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
      area of the image must contain at least this fraction of any bounding box
      supplied.
    aspect_ratio_range: An optional list of `floats`. The cropped area of the
      image must have an aspect ratio = width / height within this range.
    area_range: An optional list of `floats`. The cropped area of the image
      must contain a fraction of the supplied image within in this range.
    max_attempts: An optional `int`. Number of attempts at generating a cropped
      region of the image of the specified constraints. After `max_attempts`
      failures, return the entire image.
    scope: Optional scope for name_scope.
  Returns:
    A tuple, a 3-D Tensor cropped_image and the distorted bbox
  """
  with tf.name_scope(scope, 'distorted_bounding_box_crop', [image, mask]):
    bbox_begin, bbox_size, bbox = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        bounding_boxes=tf.constant([0.0, 0.0, 1.0, 1.0],
                         dtype=tf.float32,
                         shape=[1, 1, 4]), # Get up-to same size bounding box as original iamge
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=max_attempts,
        use_image_if_no_bounding_boxes=True)
    
    # Crop the image to the specified bounding box.
    cropped_image = tf.slice(image, bbox_begin, bbox_size)
    
    # Resize the crop back to the original image size. Resizing requires "batch" dimension.
    cropped_image = tf.expand_dims(cropped_image, 0)
    cropped_image = tf.image.resize_bilinear(cropped_image, [image_size, image_size],
                                      align_corners=False)
    # Remove batch dimension
    cropped_image = tf.squeeze(cropped_image, 0)
    
    # Crop the mask to the same bounding box we used to crop the image above.
    # Slice doesn't need the batch dimension, so remove that
    mask = tf.expand_dims(mask, 2)
    cropped_mask = tf.slice(mask, bbox_begin, bbox_size)
    
    # Resize the mask back to its original size
    # Resize requires the batch dimension
    cropped_mask = tf.expand_dims(cropped_mask, 0)
    cropped_mask = tf.image.resize_nearest_neighbor(cropped_mask, [mask_size, mask_size], align_corners=False)
    
    # Remove the batch and channel dimension from the final mask
    cropped_mask = tf.squeeze(cropped_mask, 0)
    cropped_mask = tf.squeeze(cropped_mask, 2)
    
    cropped_image.set_shape((image_size, image_size, 3))
    cropped_mask.set_shape((mask_size, mask_size))
    
    return cropped_image, cropped_mask


def preprocess_for_train(image, mask,
                         image_size, mask_size,
                         fast_mode=True,
                         scope=None,
                         add_image_summaries=True):
  with tf.name_scope(scope, 'distort_image', [image, mask]):
    if image.dtype != tf.float32:
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    cropped_image, cropped_mask = distorted_bounding_box_crop(image, mask, image_size, mask_size)

    # Randomly flip the image horizontally.
    p_order = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    pred = tf.less(p_order, 0.5)
    
    # Flip requires channel dimension, so add that to mask
    cropped_mask = tf.expand_dims(cropped_mask, 2)
    
    distorted_image = tf.cond(pred, lambda: tf.image.flip_left_right(cropped_image), lambda: cropped_image)
    distorted_mask = tf.cond(pred, lambda: tf.image.flip_left_right(cropped_mask), lambda: cropped_mask)
    
    # Remove channel dimension from mask
    distorted_mask = tf.squeeze(distorted_mask, 2)
    
    distorted_image.set_shape((image_size, image_size, 3))
    distorted_mask.set_shape((mask_size, mask_size))
    
#     # Randomly distort the colors. There are 4 ways to do it.
#     distorted_image = apply_with_random_selector(
#         distorted_image,
#         lambda x, ordering: distort_color(x, ordering, fast_mode),
#         num_cases=4)

#     distorted_image = tf.subtract(distorted_image, 0.5)
#     distorted_image = tf.multiply(distorted_image, 2.0)
    return distorted_image, distorted_mask


def preprocess_for_eval(image, image_size, central_fraction=0.875, scope=None):
  with tf.name_scope(scope, 'eval_image', [image]):
    if image.dtype != tf.float32:
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # Crop the central region of the image with an area containing 87.5% of
    # the original image.
    if central_fraction:
      cropped = tf.image.central_crop(image, central_fraction=central_fraction)

    # Resize the image back to original height/width after cropping
    # resize_bilinear needs a 4D image (batch, width, height, channel)
    cropped = tf.expand_dims(cropped, 0)
    image = tf.image.resize_bilinear(cropped, [image_size, image_size],
                                      align_corners=False)
    image = tf.squeeze(image, [0])

#     image = tf.subtract(image, 0.5)
#     image = tf.multiply(image, 2.0)
    return image


def preprocess_image(image, mask,
                     image_size, mask_size,
                     is_eval=False,
                     add_image_summaries=True):
  if is_eval:
    return preprocess_for_eval(image, image_size)
  else:
    return preprocess_for_train(image, mask, image_size, mask_size, add_image_summaries=add_image_summaries)