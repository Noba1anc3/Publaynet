import utils
import json
import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
setup_logger()

from google.colab import drive
drive.mount('/content/drive/')

trainPath = "./publaynet_data/image/"
trainjsonPath = "./publaynet_data/data.json"

with open(trainjsonPath, 'r') as f:
    imgs_anns = json.load(f)
    print('json loader finished')

images = {}
for image in imgs_anns['images']:
    images[image['id']] = {'file_name': image['file_name'], 'annotations': []}
for ann in imgs_anns['annotations']:
    images[ann['image_id']]['annotations'].append(ann)

categories = []
for img in imgs_anns['categories']:
    categories.append(img['name'])
print('categories: ', categories)

DatasetCatalog.register("trainSet", lambda I = images, P = trainPath: utils.get_textImg_dicts(I, P))
MetadataCatalog.get("trainSet").set(thing_classes=categories)
textImg_metadata = MetadataCatalog.get("trainSet")
print('textImg_metadata: ', textImg_metadata)

cfg = get_cfg()

cfg.MODEL.DEVICE = 'cpu'
cfg.OUTPUT_DIR = './output_publaynet'

cfg.merge_from_file("./detectron2/configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("trainSet",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 6
# cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x/139173657/model_final_68b088.pkl"
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.SOLVER.IMS_PER_BATCH = 8
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.MAX_ITER = 190004
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=True)
trainer.train()
print('train finished')

# cfg.save_model_steps
