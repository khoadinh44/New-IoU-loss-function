import os 
import sys
import numpy as np 
from utils import config

data_path = os.path.join(config.base_dir,config.image_dir)
imgdata = [os.path.join(data_path,img[:-4])for img in os.listdir(data_path)]

fd = open(os.path.join(config.base_dir,'train.txt'),'w+')
for filename in imgdata:
	fd.write(filename)
	fd.write('\n')
print("Images written in train.txt ",len(imgdata))

fd.close()
# copy xml from labels to images for train.py 

