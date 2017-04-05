images_dir="/Users/jt/Desktop/V2/source"
pred_file="/Users/jt/Desktop/V2/v4_basedon_v4_9395_pred_data=V2.csv"
analysis_dir="/Users/jt/Desktop/V2/analysis"
submit_orders=["ALB","BET","DOL","LAG","NoF","OTHER","SHARK","YFT"]

import os
import pandas as pd
import sys
import numpy as np
from shutil import copyfile

pred_data=pd.read_csv(pred_file,header=0,index_col=0)
pred_names=pred_data.index.values
pred_poss=pred_data.values
pred_names_dic=dict(zip(pred_names,[i for i in range(len(pred_names))]))

for label in os.listdir(images_dir):
    scr_path = os.path.join(images_dir, label)
    dst_path = os.path.join(analysis_dir, label)
    if os.path.exists(dst_path):
        raise Exception
    else:
        os.makedirs(dst_path)
    for name in os.listdir(scr_path):
        poss=pred_poss[pred_names_dic[name]]
        index=np.argmax(poss)
        index_dir=submit_orders[index]
        if os.path.exists(os.path.join(dst_path,index_dir))==False:
            os.makedirs(os.path.join(dst_path,index_dir))
        copyfile(os.path.join(scr_path,name),os.path.join(dst_path,index_dir,name))
