from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath( execution_path + "/models/detection_model-ex-003--loss-14.081.h5")
detector.loadModel()


detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "snapshot.jpg"), output_image_path=os.path.join(execution_path , "image3custom.jpg"))

for eachObject in detections:
    print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
    print("--------------------------------")