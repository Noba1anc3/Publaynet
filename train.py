
import os
import utils
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.evaluation import DatasetEvaluator, inference_on_dataset

setup_logger()

trainPath = "./data/dev/images/"
trainjsonPath = "./data/dev/dev.json"
testPath = "./data/test/images/"
testjsonPath = "./data/test/test.json"

train_images, test_images, categories = utils.json_resolve(trainjsonPath, testjsonPath)

DatasetCatalog.register("trainSet", lambda I = train_images, P = trainPath: utils.get_textImg_dicts(I, P))
MetadataCatalog.get("trainSet").set(thing_classes=categories)
textImg_metadata = MetadataCatalog.get("trainSet")
print('textImg_metadata: ', textImg_metadata)

DatasetCatalog.register("testSet", lambda I = test_images, P = testPath: utils.get_textImg_dicts(I, P))
MetadataCatalog.get("testSet").set(thing_classes=categories)
textImg_metadata = MetadataCatalog.get("testSet")
print('textImg_metadata: ', textImg_metadata)

cfg = get_cfg()

#cfg.MODEL.DEVICE = 'cpu'

cfg.OUTPUT_DIR = './output'
cfg.merge_from_file("./detectron2/configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("trainSet",)
cfg.DATASETS.TEST = ("testSet",)
cfg.DATALOADER.NUM_WORKERS = 6
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 1e-3
cfg.SOLVER.CHECKPOINT_PERIOD = 100
cfg.SOLVER.MAX_ITER = 190100
#cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x/139173657/model_final_68b088.pkl"
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=True)
trainer.train()

evaluator = DatasetEvaluator()
val_loader = build_detection_test_loader(cfg, "testSet")
inference_on_dataset(trainer.model, val_loader, evaluator)
