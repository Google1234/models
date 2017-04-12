import  tensorflow as tf
from tensorflow.python.ops import control_flow_ops
def _smallest_size_at_least(height, width, smallest_side):
  """Computes new shape with the smallest side equal to `smallest_side`.

  Computes new shape with the smallest side equal to `smallest_side` while
  preserving the original aspect ratio.

  Args:
    height: an int32 scalar tensor indicating the current height.
    width: an int32 scalar tensor indicating the current width.
    smallest_side: A python integer or scalar `Tensor` indicating the size of
      the smallest side after resize.

  Returns:
    new_height: an int32 scalar tensor indicating the new height.
    new_width: and int32 scalar tensor indicating the new width.
  """
  smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

  height = tf.to_float(height)
  width = tf.to_float(width)
  smallest_side = tf.to_float(smallest_side)

  scale = tf.cond(tf.greater(height, width),
                  lambda: smallest_side / width,
                  lambda: smallest_side / height)
  new_height = tf.to_int32(height * scale)
  new_width = tf.to_int32(width * scale)
  return new_height, new_width

def _aspect_preserving_resize(image, smallest_side):
  """Resize images preserving the original aspect ratio.

  Args:
    image: A 3-D image `Tensor`.
    smallest_side: A python integer or scalar `Tensor` indicating the size of
      the smallest side after resize.

  Returns:
    resized_image: A 3-D tensor containing the resized image.
  """
  smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

  shape = tf.shape(image)
  height = shape[0]
  width = shape[1]
  new_height, new_width = _smallest_size_at_least(height, width, smallest_side)
  image = tf.expand_dims(image, 0)
  resized_image = tf.image.resize_bilinear(image, [new_height, new_width],
                                           align_corners=False)
  resized_image = tf.squeeze(resized_image)
  resized_image.set_shape([None, None, 3])
  return resized_image

def crop_144(image,target_height,target_width):
  resizes=[299,360,480,700]# element must >=target_height and target_width,for game is 299
  #resizes=[299]
  #has a small bug: some size may cause error,for exapmle size=400 or 500,
  #to pass this,just amend slightly size,for example change size to 399 or 401
  '''
      image: 3-D Tensor of image. If dtype is tf.float32 then the range should be
      [0, 1], otherwise it would converted to tf.float32 assuming that the range
      is [0, MAX], where MAX is largest positive representable number for
      int(8/16/32) data type (see `tf.image.convert_image_dtype` for details).
  '''
  if image.dtype != tf.float32:
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  def crop(size):
    #resize image
    resized_image=_aspect_preserving_resize(tf.identity(image),size)
      
    #defalut :the resized image width>height
    shape = tf.shape(resized_image)
    height = shape[0]
    width = shape[1]
    my_assert=tf.assert_greater_equal(width,height,['My Assert:image width must >= heigth.'])    
    #left,middle,right crop (*3)
    cropped_shape = control_flow_ops.with_dependencies([my_assert],tf.pack([size, size, shape[2]]))
    left_offsets =  tf.to_int32(tf.pack([0, 0, 0]))#height,width,channel
    mid_offsets  =  tf.to_int32(tf.pack([0, (width-size)/2, 0]))
    right_offsets=  tf.to_int32(tf.pack([0, width-size, 0])) 
    postions_packs_images=tf.pack([tf.slice(resized_image, left_offsets, cropped_shape),tf.slice(resized_image, mid_offsets, cropped_shape),tf.slice(resized_image, right_offsets, cropped_shape)])
     
    #top-left,top-right,middle,bottom-left,botton-right (*6)
    cropped_shape = tf.pack([target_height, target_width, shape[2]])#height,width,channel
    top_left_offsets =  tf.to_int32(tf.pack([0, 0, 0]))#height,width,channel
    top_right_offsets =  tf.to_int32(tf.pack([0, size-target_width, 0]))#height,width,channel
    middle_offsets =  tf.to_int32(tf.pack([(size-target_height)/2, (size-target_width)/2, 0]))#height,width,channel
    bottom_left_offsets =  tf.to_int32(tf.pack([(size-target_height), 0, 0]))#height,width,channel
    bottom_right_offsets =  tf.to_int32(tf.pack([size-target_height, size-target_width, 0]))#height,width,channel 
    corners=tf.pack([top_left_offsets,top_right_offsets,middle_offsets,bottom_left_offsets,bottom_right_offsets])
    corners_packs_images=tf.pack([tf.slice(_img, _offsets, cropped_shape) for _img in tf.unpack(postions_packs_images) for _offsets in tf.unpack(corners)])
    
    middle_resize_image=tf.image.resize_images(images=postions_packs_images,size=[target_height,target_width],align_corners=False) 
    crop_postions_packs_images=tf.concat(0,[corners_packs_images,middle_resize_image]) 

    # mirros (*2)     
    _shape = tf.pack([target_height, target_width, shape[2]])#height,width,channel
    _offsets =  tf.to_int32(tf.pack([0, 0, 0]))#height,width,channel 
    mirro_packs_images=tf.pack([tf.image.flip_left_right(tf.slice(_crop, _offsets, _shape)) for _crop in tf.unpack(crop_postions_packs_images)])
    rst_packs_images=tf.concat(0,[mirro_packs_images,crop_postions_packs_images])
   
    rst_packs_images.set_shape([1*3*6*2,target_height,target_width,3])  

    #tf.image_summary('source image',      tf.expand_dims(image, 0))
    #tf.image_summary('resized_image ', tf.expand_dims(resized_image,0),144)
    #tf.image_summary('postions_packs_images ', postions_packs_images,144)
    #tf.image_summary('corners_packs_images',corners_packs_images,144)
    #tf.image_summary('crop_postions_packs_images',crop_postions_packs_images,144)
    #tf.image_summary('mirro_packs_images',mirro_packs_images,144)
    #tf.image_summary('mirro_packs_images',rst_packs_images,144)
    #print ('###########################') 
    #print ('total crops:',rst_packs_images)
    s="size:"+str(size)
    tf.image_summary(s,rst_packs_images,1*3*6*2)
    return rst_packs_images
  rst=tf.concat(0,[crop(_size) for _size in resizes])
  rst.set_shape([len(resizes)*3*6*2,target_height,target_width,3])
  print ('###########################')
  print (rst)
  tf.image_summary('source_image',tf.expand_dims(image,0),144)
  tf.image_summary('rst_crops_images',rst,144)
  return rst
def crop_1(image,target_height,target_width):
  '''
      image: 3-D Tensor of image. If dtype is tf.float32 then the range should be
      [0, 1], otherwise it would converted to tf.float32 assuming that the range
      is [0, MAX], where MAX is largest positive representable number for
      int(8/16/32) data type (see `tf.image.convert_image_dtype` for details).
  '''
  if image.dtype != tf.float32:
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  tf.image_summary('source_image',tf.expand_dims(image,0),144)
  resized_image=tf.image.resize_bilinear(tf.expand_dims(image,0), [target_height, target_width],
                                           align_corners=True)
  tf.image_summary('resized_image',resized_image,144)
  return resized_image










    


































 
  Dprint (rst)
  tf.image_summary('source_image',tf.expand_dims(image,0),144)
  #tf.image_summary('rst_crops_images',rst,144)
  return rst
