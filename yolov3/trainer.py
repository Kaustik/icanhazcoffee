from imageai.Detection.Custom import DetectionModelTrainer
import os

execution_path = os.getcwd()

trainer = DetectionModelTrainer()
trainer.setGpuUsage(1)
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory=".")
trainer.setTrainConfig(object_names_array=["empty coffee", "not empty coffee"], batch_size=4, num_experiments=10,train_from_pretrained_model="pretrained-yolov3.h5")
trainer.trainModel()