from os.path import join
import numpy as np

epochs = 5
batch_size = 16
image_size = 640
base_dir = 'E:/Yolov5_voc/data/'
image_dir = 'all_images/'
label_dir = 'all_labels/'
threshold = 0.4
max_boxes = 100

class_names = {0: 'person',
             1: 'bird',
             2: 'cat',
             3: 'cow',
             4: 'dog',
             5: 'horse',
             6: 'sheep',
             7: 'aeroplane',
             8: 'bicycle',
             9: 'boat',
             10: 'bus',
             11: 'car',
             12: 'motorbike',
             13: 'train',
             14: 'bottle',
             15: 'chair',
             16: 'diningtable',
             17: 'pottedplant',
             18: 'sofa',
             19: 'tvmonitor'}
class_dict = {'person': 0,
             'bird': 1,
             'cat': 2,
             'cow': 3,
             'dog': 4,
             'horse': 5,
             'sheep': 6,
             'aeroplane': 7,
             'bicycle': 8,
             'boat': 9,
             'bus': 10,
             'car': 11,
             'motorbike': 12,
             'train': 13,
             'bottle': 14,
             'chair': 15,
             'diningtable': 16,
             'pottedplant': 17,
             'sofa': 18,
             'tvmonitor': 19}
anchors = np.array([[ 15.,  22.],
                     [ 36.,  37.],
                     [ 43.,  75.],
                     [ 66., 144.],
                     [111.,  83.],
                     [121., 204.],
                     [198., 288.],
                     [277., 155.],
                     [376., 327.]], np.float32)
                         
weights_path = 'E:/Yolov5_voc/weights/'
best_weights = 'model_s.h5'

width = [0.50, 0.75, 1.0, 1.25]
depth = [0.33, 0.67, 1.0, 1.33]

versions = ['s', 'm', 'l', 'x']
version = 's'



