
import os

from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

setup_logger()

trainPath = "./data/val/images/"
trainjsonPath = "./data/val/val.json"
testPath = "./data/test/images/"
testjsonPath = "./data/test/test.json"

register_coco_instances("trainSet", {}, trainjsonPath, trainPath)
register_coco_instances("testSet", {}, testjsonPath, testPath)

cfg = get_cfg()

# cfg.MODEL.DEVICE = 'cpu'

cfg.OUTPUT_DIR = '../drive/'My Drive'/output
cfg.merge_from_file("./detectron2/configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("trainSet",)
# cfg.DATASETS.TEST = ("testSet",)
cfg.DATALOADER.NUM_WORKERS = 6

cfg.SOLVER.CHECKPOINT_PERIOD = 100
cfg.SOLVER.MAX_ITER = 100
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 1e-3

cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x/139173657/model_final_68b088.pkl"
#cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=True)

for i in range(10):
    trainer.train()
    evaluator = COCOEvaluator("testSet", cfg, False, output_dir = cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "testSet")
    inference_on_dataset(trainer.model, val_loader, evaluator)
    cfg.SOLVER.MAX_ITER += 100
