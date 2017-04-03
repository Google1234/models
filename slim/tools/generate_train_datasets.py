# merge good image datasets with trainCrop dataset
input_images_dir="/Users/jt/Desktop/V2/trainCrop/"
good_images_dir="/Users/jt/Desktop/V2/amend_for_train/good/"
output_images_dir="/Users/jt/Desktop/V2/add_good/"
add_good_times=2

labels=["ALB","BET","DOL","LAG","NoF","OTHER","SHARK","YFT"]
#add good images

import os
import sys
from PIL import  Image
for label in labels:
    print ("generate ",label)

    if label=="NoF":
        if len(os.listdir(os.path.join(input_images_dir,label)))<3:
            print ("NoF dir has no iamges ,may be run python generate_NoF_images.py before")
            raise  Exception
    if os.path.exists(os.path.join(output_images_dir,label))==True:
        print ("path :",os.path.join(output_images_dir,label)," already exist")
        raise Exception
    os.mkdir(os.path.join(output_images_dir,label))
    for name in os.listdir(os.path.join(input_images_dir,label)): 
        t = 1
        img=Image.open(os.path.join(input_images_dir,label,name))
        img.save(os.path.join(output_images_dir, label, name[:-4] + "_" + str(t) + ".jpg"))
        t+=1
        left_right=img.transpose(Image.FLIP_LEFT_RIGHT)
        up_down = img.transpose(Image.FLIP_TOP_BOTTOM)
        left_right.save(os.path.join(output_images_dir,label,name[:-4]+"_"+str(t)+".jpg"))
        t+=1
        up_down.save(os.path.join(output_images_dir,label,name[:-4]+"_"+str(t)+".jpg"))
        t+=1
    if label=="NoF":
        continue
    for name in os.listdir(os.path.join(good_images_dir,label)):
        t=4 
        img=Image.open(os.path.join(good_images_dir,label,name))
        left_right=img.transpose(Image.FLIP_LEFT_RIGHT)
        up_down = img.transpose(Image.FLIP_TOP_BOTTOM)
        for add in range(add_good_times):
            img.save(os.path.join(output_images_dir, label, name[:-4] + "_" + str(t) + ".jpg"))
            t+=1
            left_right.save(os.path.join(output_images_dir,label,name[:-4]+"_"+str(t)+".jpg"))
            t+=1
            up_down.save(os.path.join(output_images_dir,label,name[:-4]+"_"+str(t)+".jpg"))
            t+=1


