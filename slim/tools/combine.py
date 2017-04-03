import re
import os
import pandas

submit_orders=["ALB","BET","DOL","LAG","NoF","OTHER","SHARK","YFT"]
class_nums=8

crop_predict_file="data/v2_6648_predcit.csv"
model2_predict_file="data/9981_144_summit.csv"
rst_file=crop_predict_file[:-4]+"_combine.csv"
image_dir="/Users/jt/code/MachineLearning/fish_data/sourcedata/test_stg1"
images_nums=1000

#input data has been softmax
'''
#my model index:label
0:ALB
1:BET
2:DOL
3:LAG
4:NoF
5:OTHER
6:SHARK
7:YFT
submit command order:image,ALB,BET,DOL,LAG,NoF,OTHER,SHARK,YFT
'''

# load use crop (small size ) model predict result
crop_data=pandas.read_csv(crop_predict_file,header=0,index_col=0)
crop_names=crop_data.index.values
crop_poss=crop_data.values
crop_names_dic=dict(zip(crop_names,[i for i in range(len(crop_names))]))

# load dont use crop (small size ) model predict result
model2_data=pandas.read_csv(model2_predict_file,header=0,index_col=0)
model2_names=model2_data.index.values
model2_poss=model2_data.values
model2_names_dic=dict(zip(model2_names,[i for i in range(len(model2_names))]))

#load image names by image_dir

names=os.listdir(image_dir)
if len(names)!=images_nums:
    print ("image dir :",image_dir," has ",len(names),'images')
    raise Exception

rst=[]
for name in names:
    if crop_names_dic.has_key(name):
        rst.append(crop_poss[crop_names_dic[name]])
    elif crop_names_dic.has_key(name[:-4]+"_1.jpg"):
        index=1
        rst.append([0.0 for i in range(class_nums)])
        while crop_names_dic.has_key(name[:-4]+"_"+str(index)+".jpg"):
            rst[-1]+=crop_poss[crop_names_dic[name[:-4]+"_"+str(index)+".jpg"]]
            index+=1
        # avr
        rst[-1]=[rst[-1][i]/(index-1) for i in range(class_nums)]
    else:
        rst.append(model2_poss[model2_names_dic[name]])

# out to csv
data=pandas.DataFrame(rst,index=names,columns=submit_orders)
#print data
data.to_csv(rst_file,index_label="image")


