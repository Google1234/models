NoF_images_path="/ssd/lidenghui/jt/fishes/train/NoF/"
save_dir="/ssd/lidenghui/jt/fishes/V2/trainCrop/NoF"
make_nums=465 # actualy get [make_nums/len(iamges)]* len(images) generate make_nums cropd_images from NoF_images_path
width_min=50
width_max=500
ratio_min=0.5
ratio_max=1.5

import os
import sys
import random
from PIL import Image
import shutil

images=os.listdir(NoF_images_path)
print ("NoF iamges nums:",len(images))
if len(os.listdir(save_dir))>0:
    print ("del already crop images ",len(images))
    shutil.rmtree(save_dir)
    os.mkdir(save_dir)
count=0
iter_t=0
while count<make_nums:
    iter_t+=1
    for name in images:
        img=Image.open(os.path.join(NoF_images_path,name))
        width, height = img.size
        crop_width = random.randint(width_min, min(width_max,width))
        crop_height = min(int(crop_width * random.uniform(ratio_min, ratio_max)),height)
        x = random.randint(0, width - crop_width)
        y = random.randint(0, height - crop_height)
        rst = img.crop((x, y, x+crop_width, y+crop_height))
        rst.save(os.path.join(save_dir,name+"_"+str(iter_t)+".jpg"))

        count+=1
print ("crop images nums:",count)

