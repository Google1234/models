from PIL import Image
import os
rotate_train=True # when rotate train image==>have dirs
source_dir="/Users/jt/Desktop/V2/add_good/"
dst_dir="/Users/jt/Desktop/V2/rotate/photos"
#rotate_train=False # when rotate test image==>no dirs
#source_dir="/ssd/lidenghui/jt/fishes/testCrop/"
#dst_dir="/ssd/lidenghui/jt/fishes/rotate_testCrop/photos"
if os.path.exists(dst_dir)==False:
  os.makedirs(dst_dir)
if rotate_train:
  for dir in os.listdir(source_dir): 
    src_path= os.path.join(source_dir,dir)
    dst_path= os.path.join(dst_dir,dir)
    if os.path.exists(dst_path)==False:
      os.mkdir(dst_path)
    for file in os.listdir(src_path): 
      img=Image.open(os.path.join(src_path,file))
      weight,height=img.size
      # let weight > height
      if weight<height:
        old_img=img.transpose(Image.ROTATE_90)
      else:
        old_img=img
      # pad to square
      old_size=old_img.size
      new_size=(old_size[0],old_size[0]) # width ,the bigger one
      new_im = Image.new("RGB", new_size)
      new_im.paste(old_img, ((new_size[0]-old_size[0])/2,(new_size[1]-old_size[1])/2))
      new_im.save(os.path.join(dst_path,file))
else:
  for file in os.listdir(source_dir): 
      img=Image.open(os.path.join(source_dir,file))
      weight,height=img.size
      # let weight > height
      if weight<height:
        old_img=img.transpose(Image.ROTATE_90)
      else:
        old_img=img
      # pad to square
      old_size=old_img.size
      new_size=(old_size[0],old_size[0])
      new_im = Image.new("RGB", new_size)
      new_im.paste(old_img, ((new_size[0]-old_size[0])/2,(new_size[1]-old_size[1])/2))
      new_im.save(os.path.join(dst_dir,file))
