import os
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    import sys
    stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    from imageai.Detection.Custom import CustomObjectDetection
    sys.stderr = stderr

def newest(path):
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    return max(paths, key=os.path.getctime)

execution_path = os.getcwd()

detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(detection_model_path=execution_path + "/models/detection_model-ex-004.h5")
detector.setJsonPath(configuration_json=execution_path + "/json/detection_config.json")
detector.loadModel()

latest_image = newest("/home/ftp")

detections = detector.detectObjectsFromImage(latest_image, minimum_percentage_probability=60, output_image_path="image-new.jpg")

for detection in detections:
    if (detection["name"] == "not empty coffee"):
        print("Det är", round(detection["percentage_probability"], 2), "% chans att det finns kaffe!")
    else:
        print("Mest troligt (", detection["percentage_probability"], ") finns det inget kaffe :(")
