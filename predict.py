import utils
import json
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

valPath = "./publaynet_data/image/"
valjsonPath = "./publaynet_data/data.json"

with open(valjsonPath, 'r') as f:
    print('loading json......')
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

DatasetCatalog.register("valSet", lambda I = images, P = valPath: utils.get_textImg_dicts(I, P))
MetadataCatalog.get("valSet").set(thing_classes=categories)
textImg_metadata = MetadataCatalog.get("valSet")
print('textImg_metadata: ', textImg_metadata)

cfg = get_cfg()

cfg.MODEL.DEVICE = 'cpu'
cfg.OUTPUT_DIR = './output_publaynet'

cfg.merge_from_file("./detectron2_repo/configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("trainSet",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 6
cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x/139173657/model_final_68b088.pkl"
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.01
cfg.SOLVER.MAX_ITER =180000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5

cfg.merge_from_file("./detectron2_repo/configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
# cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg.DATASETS.TEST = ("valSet", )

predictor = DefaultPredictor(cfg)
print('predictor ready')

'''
def timekeeping(predictor, test_num):
    time_used_total = 0
    for d in random.sample(utils.get_textImg_dicts(images, valPath), test_num):
        img = cv2.imread(d["file_name"])
        time_before_pred = time.time()
        output = predictor(img)
        time_used_total += time.time()-time_before_pred
    print('{0} s per sample when inferencing.'.format(time_used_total/test_num))
'''

def eval_visualization(predictor):
    utils.draw_predImg_dicts(
        utils.get_textImg_dicts(images, valPath), 10, textImg_metadata, predictor)

if __name__ == '__main__':
    # timekeeping(predictor, 100)
    eval_visualization(predictor)
