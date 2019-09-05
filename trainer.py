from imageai.Detection.Custom import DetectionModelTrainer
import os

execution_path = os.getcwd()

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory=execution_path)
trainer.setTrainConfig(object_names_array=["empty coffee", "not empty coffee"], batch_size=4, num_experiments=4)
trainer.trainModel()