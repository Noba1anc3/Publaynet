import os
import random
from detectron2.structures import BoxMode
import cv2
from detectron2.utils.visualizer import Visualizer
from matplotlib import pyplot as plt
import utils
import json
import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

def draw_textImg_dicts(dataset_dicts, smp_num, textImg_metadata):
    for d in random.sample(dataset_dicts, smp_num):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=textImg_metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        img = vis.get_image()[:, :, ::-1]
        plt.imshow(img)
        plt.show()
