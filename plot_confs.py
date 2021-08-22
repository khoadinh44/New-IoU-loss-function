from import_file import import_file
confusionMatrix = import_file('E:/Yolov5_Tensorflow_main/ConfusionMatrix.py')

def main():
    if not exists('results'):
        os.makedirs('results')
    f_names = []
    with open(join(config.base_dir, 'val.txt')) as reader:
        lines = reader.readlines()
    for line in lines:
        f_names.append(line.rstrip().split(' ')[0])
    result_dict = {}

    model = nn.build_model(training=False)
    model.load_weights("E:/Yolov5_Tensorflow_main/weights/model21.h5", True)

    for f_name in tqdm.tqdm(f_names):
        image_path = join(config.base_dir, config.image_dir, f_name + '.png')
        label_path = join(config.base_dir, config.label_dir, f_name + '.xml')
        image = cv2.imread(image_path)
        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_np, scale, dw, dh = util.resize(image_np)
        image_np = image_np.astype(np.float32) / 255.0

        boxes, scores, labels = model.predict(image_np[np.newaxis, ...])
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / scale

        confusion_matrix = ConfusionMatrix(nc=nc)
        confusion_matrix.process_batch(predn, torch.cat((labels[:, 0:1], tbox), 1))

        