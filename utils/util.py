from os.path import join
from xml.etree.ElementTree import ParseError
from xml.etree.ElementTree import parse as parse_fn
import os
import cv2
import numpy as np
from six import raise_from
from import_file import import_file
# config = import_file('E:/Yolov5_voc/utils/config.py')
from utils import config

def find_node(parent, name, debug_name=None, parse=None):
    if debug_name is None:
        debug_name = name

    result = parent.find(name)
    if result is None:
        raise ValueError(f'missing element \'{debug_name}\'')
    if parse is not None:
        try:
            return parse(result.text)
        except ValueError as e:
            raise_from(ValueError(f'illegal value for \'{debug_name}\': {e}'), None)
    return result


def parse_annotation(element):
    # truncated = find_node(element, 'truncated', parse=int)
    # difficult = find_node(element, 'difficult', parse=int)

    class_name = find_node(element, 'name').text
    if class_name not in config.class_dict:
        raise ValueError(f'class name \'{class_name}\' not found in class_dict: {list(config.class_dict.keys())}')

    label = config.class_dict[class_name]

    box = find_node(element, 'bndbox')
    x_min = find_node(box, 'xmin', 'bndbox.xmin', parse=float)
    y_min = find_node(box, 'ymin', 'bndbox.ymin', parse=float)
    x_max = find_node(box, 'xmax', 'bndbox.xmax', parse=float)
    y_max = find_node(box, 'ymax', 'bndbox.ymax', parse=float)

    # return truncated, difficult, [x_min, y_min, x_max, y_max], label
    return [x_min, y_min, x_max, y_max], label


def parse_annotations(xml_root):
    boxes = []
    labels = []
    for i, element in enumerate(xml_root.iter('object')):
        # truncated, difficult, box, label = parse_annotation(element)
        box, label = parse_annotation(element)
        boxes.append(box)
        labels.append(label)

    boxes = np.asarray(boxes, np.float32)
    labels = np.asarray(labels, np.int32)
    return boxes, labels


def load_image(f_name):
    image=[]
    path = join(config.base_dir, config.image_dir, f_name+'.jpg')
    if os.path.isfile(path):
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def load_label(f_name):
    try:
        tree = parse_fn(join(config.base_dir, config.label_dir, f_name + '.xml'))
        return parse_annotations(tree.getroot())
    except ParseError as error:
        raise_from(ValueError(f'invalid annotations file: {f_name}: {error}'), None)
    except ValueError as error:
        raise_from(ValueError(f'invalid annotations file: {f_name}: {error}'), None)


def random_horizontal_flip(image, boxes):
    if np.random.random() > 0.8:
        _, w, _ = image.shape
        image = image[:, ::-1, :]
        boxes[:, [0, 2]] = w - boxes[:, [2, 0]]

    return image, boxes


def random_noise(image):
    if np.random.random() > 0.6:
        image = cv2.GaussianBlur(image, (5, 5), np.random.uniform(0, 2))
    return image


def resize(image, boxes=None):
    h, w, _ = image.shape

    scale = min(config.image_size / w, config.image_size / h)
    w = int(scale * w)
    h = int(scale * h)

    image_resized = cv2.resize(image, (w, h))

    image_padded = np.zeros(shape=[config.image_size, config.image_size, 3], dtype=np.uint8)
    dw, dh = (config.image_size - w) // 2, (config.image_size - h) // 2
    image_padded[dh:h + dh, dw:w + dw, :] = image_resized.copy()

    if boxes is None:
        return image_padded, scale, dw, dh

    else:
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + dw
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + dh

        return image_padded, boxes


def process_box(boxes, labels):
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    anchors = config.anchors
    box_centers = (boxes[:, 0:2] + boxes[:, 2:4]) / 2
    box_size = boxes[:, 2:4] - boxes[:, 0:2]

    y_true_1 = numpy.zeros((config.image_size // 32,
                            config.image_size // 32,
                            3, 5 + len(config.class_dict)), numpy.float32)
    y_true_2 = numpy.zeros((config.image_size // 16,
                            config.image_size // 16,
                            3, 5 + len(config.class_dict)), numpy.float32)
    y_true_3 = numpy.zeros((config.image_size // 8,
                            config.image_size // 8,
                            3, 5 + len(config.class_dict)), numpy.float32)

    y_true = [y_true_1, y_true_2, y_true_3]

    box_size = numpy.expand_dims(box_size, 1)

    min_np = numpy.maximum(- box_size / 2, - anchors / 2)
    max_np = numpy.minimum(box_size / 2, anchors / 2)

    whs = max_np - min_np

    overlap = whs[:, :, 0] * whs[:, :, 1]
    union = box_size[:, :, 0] * box_size[:, :, 1] + anchors[:, 0] * anchors[:, 1] - whs[:, :, 0] * whs[:, :, 1] + 1e-10

    iou = overlap / union
    best_match_idx = numpy.argmax(iou, axis=1)

    ratio_dict = {1.: 8., 2.: 16., 3.: 32.}
    for i, idx in enumerate(best_match_idx):
        feature_map_group = 2 - idx // 3
        ratio = ratio_dict[numpy.ceil((idx + 1) / 3.)]
        x = int(numpy.floor(box_centers[i, 0] / ratio))
        y = int(numpy.floor(box_centers[i, 1] / ratio))
        k = anchors_mask[feature_map_group].index(idx)
        c = labels[i]

        y_true[feature_map_group][y, x, k, :2] = box_centers[i]
        y_true[feature_map_group][y, x, k, 2:4] = box_size[i]
        y_true[feature_map_group][y, x, k, 4] = 1.
        y_true[feature_map_group][y, x, k, 5 + c] = 1.

    return y_true_1, y_true_2, y_true_3
