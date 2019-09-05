from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory=".")
trainer.setTrainConfig(object_names_array=["empty coffee", "not empty coffee"], batch_size=4, num_experiments=4)
trainer.trainModel()