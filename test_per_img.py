import os
import cv2
import time
import numpy as np
import matplotlib
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from matplotlib import pyplot as plt

time_used = []
test_data_dir = './data/test/images'
test_data_pred_dir = './prediction/'
files = os.listdir(test_data_dir)

categories = ['table', 'list', 'title', 'text', 'figure']
print('categories: ', categories)

# DatasetCatalog.register("valSet", lambda I = images, P = valPath: utils.get_textImg_dicts(I, P))
MetadataCatalog.get("valSet").set(thing_classes=categories)
textImg_metadata = MetadataCatalog.get("valSet")

print('textImg_metadata: ', textImg_metadata)

cfg = get_cfg()

cfg.merge_from_file("./detectron2/configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
cfg.DATALOADER.NUM_WORKERS = 6
cfg.MODEL.DEVICE = 'cpu'
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg.MODEL.WEIGHTS = os.path.join('./output', "model_final.pth")

predictor = DefaultPredictor(cfg)

for file in files:
    print(os.path.join(test_data_dir, file))
    im = cv2.imread(os.path.join(test_data_dir, file))

    start_time = time.time()
    outputs = predictor(im)
    time_used.append(time.time()-start_time)

    print(outputs["instances"].pred_classes)
    print(outputs["instances"].pred_boxes + '\n')

    v = Visualizer(im[:, :, ::-1], textImg_metadata, scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    img = v.get_image()[:, :, ::-1]

    # plt.imshow(img)
    # plt.show()
    # matplotlib.image.imsave(os.path.join(test_data_pred_dir, file), img)

print('time used per img: {0} sec'.format(np.mean(np.array(time_used))))
