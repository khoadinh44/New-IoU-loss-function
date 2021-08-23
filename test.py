import os
import pickle
from os.path import exists
from os.path import join
from xml.etree.ElementTree import parse as parse_fn
import cv2
import numpy as np
import tqdm
from import_file import import_file
import matplotlib
import matplotlib.pyplot as plt

import ConfusionMatrix
from utils import config
from nets import nn
from utils import util
# confusionMatrix = import_file('E:/Yolov5_voc/ConfusionMatrix.py')
# config = import_file('E:/Yolov5_voc/utils/config.py')
# nn = import_file('E:/Yolov5_voc/nets/nn.py')
# util = import_file('E:/Yolov5_voc/utils/util.py')
matplotlib.use("Agg")

def test():
    graphic = ConfusionMatrix.ConfusionMatrix(42, conf=0.7, iou_thres=0.7)
    def draw_bbox(image, boxes, labels, scores, true_boxes, true_labels, file_name):
        detection = []
        txt = open('mAP-master/input/detection-results/{}.txt'.format(file_name), 'w')
        for i, box in enumerate(boxes):
            coordinate = np.array(box[:4], dtype=np.int32)
            c1, c2 = (coordinate[0], coordinate[1]), (coordinate[2], coordinate[3])
            cv2.rectangle(image, c1, c2, (255, 0, 0), 1)

            # confusion matrix==============================================
            detection.append([c1[0], c1[1], c2[0], c2[1], scores[i], labels[i]])

            # write txt true
            txt_true = open('mAP-master/input/ground-truth/{}.txt'.format(file_name), 'w')
            for idx, true_box in enumerate(true_boxes):
                txt_true.write(f'{true_labels[idx]} {true_box[0]} {true_box[1]} {true_box[2]} {true_box[3]}\n')
            txt_true.close()

            ind = labels[i]
            # print('predict: ', true_boxes)
            # print('true labels: ', true_labels)
            if ind != -1:
                txt.write('{} {} {} {} {} {}\n'.format(int(ind), scores[i], coordinate[0], coordinate[1], coordinate[2], coordinate[3]))
                cv2.putText(image, '{}: {}%'.format(config.class_names[ind], float(scores[i]*100)),(c1[0], c1[1]-2), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
            
            # clear window when training run --------------------------------------------------------------
            plt.cla()
            plt.clf()
            plt.close('all')

        txt.close()
        true = np.concatenate((true_labels.reshape(-1, 1), true_boxes), axis=1)
        detection = np.array(detection)
        graphic.process_batch(detection, true)
        return image

    def test_fn():
        if not os.path.exists('results'):
            os.makedirs('results')
        file_names = []
        with open(os.path.join(config.base_dir, 'val.txt')) as f:
            for file_name in f.readlines():
                file_names.append(file_name.rstrip())

        # file name ----------------------------------------------------------------------------------------
        # file_names = file_names[1134:]
        model = nn.build_model(training=False)
        model.load_weights(f"{config.weights_path + config.best_weights}", True)

        for file_name in tqdm.tqdm(file_names):
            image = cv2.imread(os.path.join(config.base_dir, config.image_dir, file_name + '.jpg'))
            image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image_np, scale, dw, dh = util.resize(image_np)
            image_np = image_np.astype(np.float32) / 255.0

            boxes, scores, labels = model.predict(image_np[np.newaxis, ...])

            boxes, scores, labels = np.squeeze(boxes, 0), np.squeeze(scores, 0), np.squeeze(labels, 0)

            boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / scale
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / scale

            true_boxes, true_labels = util.load_label(file_name)
            image = draw_bbox(image, boxes, labels, scores, true_boxes, true_labels, file_name)
            # cv2.imwrite(f'results/{file_name}.png', image)
            graphic.plot()
    test_fn()
test()
