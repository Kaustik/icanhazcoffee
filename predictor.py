from imageai.Detection.Custom import CustomObjectDetection
import os
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

def newest(path):
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    return max(paths, key=os.path.getctime)

execution_path = os.getcwd()

detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(detection_model_path="models/detection_model-ex-004.h5")
detector.setJsonPath(configuration_json="json/detection_config.json")
detector.loadModel()

latest_image = newest("/home/ftp")
#latest_image = "snapshot.jpg"

detections = detector.detectObjectsFromImage(latest_image, minimum_percentage_probability=60, output_image_path="image-new.jpg")

for detection in detections:
    print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])


