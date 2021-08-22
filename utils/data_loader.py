import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import OrderedEnqueuer
from tensorflow.keras.utils import Sequence
from import_file import import_file
# config = import_file('E:/Yolov5_voc/utils/config.py')
# util = import_file('E:/Yolov5_voc/utils/util.py')
from utils import config
from nets import nn

class Generator(Sequence):
    def __init__(self, f_names):
        self.f_names = f_names

    def __len__(self):
        return int(np.floor(len(self.f_names) / config.batch_size))

    def __getitem__(self, index):
        image = load_image(self.f_names[index])
        boxes, label = load_label(self.f_names[index])
        boxes = np.concatenate((boxes, np.full(shape=(boxes.shape[0], 1), fill_value=1., dtype=np.float32)), axis=-1)

        image = random_noise(image)

        image, boxes = random_horizontal_flip(image, boxes)
        image, boxes = resize(image, boxes)

        image = image.astype(np.float32)
        image = image / 255.0
        y_true_1, y_true_2, y_true_3 = process_box(boxes, label)
        return image, y_true_1, y_true_2, y_true_3

    def on_epoch_end(self):
        np.random.shuffle(self.f_names)


def input_fn(f_names):
    def generator_fn():
        generator = OrderedEnqueuer(Generator(f_names), True)
        generator.start(workers=8, max_queue_size=10)
        while True:
            image, y_true_1, y_true_2, y_true_3 = generator.get().__next__()
            yield image, y_true_1, y_true_2, y_true_3

    output_types = (tf.float32, tf.float32, tf.float32, tf.float32)
    output_shapes = ((config.image_size, config.image_size, 3),
                     (config.image_size // 32, config.image_size // 32, 3, len(config.class_dict) + 5),
                     (config.image_size // 16, config.image_size // 16, 3, len(config.class_dict) + 5),
                     (config.image_size // 8, config.image_size // 8, 3, len(config.class_dict) + 5),)

    dataset = tf.data.Dataset.from_generator(generator=generator_fn,
                                             output_types=output_types,
                                             output_shapes=output_shapes)

    dataset = dataset.repeat(config.epochs + 1)
    dataset = dataset.batch(config.batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset
