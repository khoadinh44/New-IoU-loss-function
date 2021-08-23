# New-IoU-loss-function

# Download data and pretraining:
Data: https://drive.google.com/drive/folders/14uaGj0u8O9aVvaiFCU5IjqrlRI0j5aN2?usp=sharing

Pretrain: https://drive.google.com/drive/folders/1dQ0DGvjEPUU5oBun9ZpmnV_sgPfdnSHf?usp=sharing


# Preparing for the test:
1. Download data and put it to the data folder, following the path:
      # Parent folder
          |__Yolov5_voc
                 |__data
                      |__all_images
                      |__all_labels
                      |__val.txt
2. Download the pretrain files and put it into the ***weights*** folder

# Test data with each weight:
  Start testing period-------------------------------------------------
  - Change **best_weights = model_CD_IoU_s_5.h**(following the weight's name) and **version = 's'** in YOLOv5/utils/config.py file
  - Run test.py
  - Run main.py in Yolov5_voc/mAP-master/main.py
 
 End training period---------------------------------------------------
   + Yolov5_voc/mAP-master/input/
        - Change the old folder (**detection-results** and **ground-truth**) to the new folder(**detection-results-model_CD_IoU_s_5** and **ground-truth-model_CD_IoU_s_5**)(following the weight's name)
        - Create the new **detection-results** and **ground-truth**
   + Yolov5_voc/mAP-master/
        - Change the old folder (**output**) to the new folder(**output-model_CD_IoU_s_5**)(following the weight's name)
        - Create the new **output** folder
