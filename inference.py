import cv2
import os
import glob
import random
import yaml
import time
import detectron2
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch, torchvision
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

def inference(config_file):
  
  cfg = get_cfg()
  cfg.MODEL.DEVICE = 'cpu'
  model_name = config_file['Model_information']['Model_name']
  if model_name == 'FASTER-RCNN':
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = config_file['Weights']['faster-rnn_weights']
    cfg.OUTPUT_DIR = config_file['Directory']['Output_dir']
    
  else: 
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_101_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = config_file['Weights']['faster-rnn_weights']
    cfg.OUTPUT_DIR = config_file['Directory']['Output_dir']

  cfg.MODEL.NMS_THRESH_TEST = config_file['Model_information']['NMS']
  cfg.MODEL.RETINANET.NMS_THRESH_TEST = config_file['Model_information']['NMS']
  cfg.MODEL.RETINANET.SCORE_THRESH_TEST = config_file['Model_information']['Threshold']
  
  cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2 
  cfg.MODEL.RETINANET.NUM_CLASSES = 2
  cfg.DATASETS.TEST = ("my_dataset_test", )
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config_file['Model_information']['Threshold'] 
  predictor = DefaultPredictor(cfg)
  test_metadata = MetadataCatalog.get("my_dataset_test")
  df = pd.DataFrame(columns=('Image_Name', 'Counts'))
  image_name=[]
  counts=[]
  num=0
  start_time = time. time()
  input_path =  config_file['Directory']['Input_dir'] + "/" +"*" + config_file['Model_information']['image_format']
  output_path = config_file['Directory']['Output_dir']
  print(input_path)
  for imageName in tqdm(glob.glob(input_path)):
    num+=1
    im = cv2.imread(imageName) 
    name = imageName.split('/')[-1]
    print(name)
    save_path =  output_path +"/"+ imageName.split('/')[-1]
    
    imgheight, imgwidth =im.shape[0],im.shape[1]
    s=0
    if imgheight > 800:
      blank_image = np.zeros((imgheight,imgwidth,3), np.uint8)
      height= int(imgheight//2)
      width= int(imgwidth//2)
      
      for i in range(0,imgheight,height):
        for j in range(0,imgwidth,width):
          
          box = (j, i, j+width, i+height)
          a = im[i:i+height,j:j+width]
          newsize=(width,height)
          outputs = predictor(a)
          v = Visualizer(a[:, :, ::-1],
                    metadata=test_metadata, 
                    scale=0.8
                    )
          for box in outputs["instances"].pred_boxes.to('cpu'):
            v.draw_box(box)
            
          v = v.get_output()
          s += len(outputs['instances'])
          blank_image[i:i+height,j:j+width]=cv2.resize(v.get_image()[:, :, ::-1], newsize, interpolation = cv2.INTER_AREA)
        
    else:
          blank_image = np.zeros((imgheight,imgwidth,3), np.uint8)
          a = cv2.resize(im, (896,896), interpolation = cv2.INTER_AREA)
          outputs = predictor(a)
          v = Visualizer(a[:, :, ::-1],
                    metadata=test_metadata, 
                    scale=0.8
                    )
          out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
          s += len(outputs['instances'])
          blank_image = cv2.resize(out.get_image()[:, :, ::-1], (600,600), interpolation = cv2.INTER_AREA)
    
    image_name.append(name)
    counts.append(s) 
    cv2.imwrite(save_path,blank_image)
  current_time = time. time()
  elapsed_time = current_time - start_time
  print("Time taken: ", elapsed_time)
    
  df['Image_Name'] = image_name
  df['Counts']=counts
  excel_path = config_file['Directory']['Output_dir'] +"/"+ "Microglia_count.xlsx"
  df.to_excel(excel_path)


if __name__ == "__main__":
    with open('config.yaml') as f:
        config_file = yaml.load(f, Loader = yaml.FullLoader) 
    inference(config_file)
  

