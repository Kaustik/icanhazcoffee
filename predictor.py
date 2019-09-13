import os
import warnings
import random

from dotenv import load_dotenv
load_dotenv()

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    import sys
    stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    from imageai.Detection.Custom import CustomObjectDetection
    sys.stderr = stderr

def get_latest_image(dirpath, valid_extensions=('jpg','jpeg','png')):
    """
    Get the latest image file in the given directory
    """
    # get filepaths of all files and dirs in the given dir
    valid_files = [os.path.join(dirpath, filename) for filename in os.listdir(dirpath)]
    # filter out directories, no-extension, and wrong extension files
    valid_files = [f for f in valid_files if '.' in f and \
        f.rsplit('.',1)[-1] in valid_extensions and os.path.isfile(f)]

    if not valid_files:
        raise ValueError("No valid images in %s" % dirpath)

    return max(valid_files, key=os.path.getmtime)

execution_path = os.path.dirname(os.path.abspath(__file__))

detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(detection_model_path=execution_path + "/yolov3/models/detection_model-ex-009--loss-8.843.h5")
detector.setJsonPath(configuration_json=execution_path + "/yolov3/json/detection_config.json")
detector.loadModel()

latest_image = get_latest_image(os.getenv("IMAGE_PATH"))
save_predictions_path = os.getenv("OUTPUT_PATH")
output_image_path = save_predictions_path + "/" + os.path.basename(latest_image)

detections = detector.detectObjectsFromImage(
    latest_image,
    minimum_percentage_probability=60,
    output_image_path=output_image_path)

for detection in detections:
    if (detection["name"] == "not empty coffee"):
        print("Det är", round(detection["percentage_probability"], 2), "% chans att det finns kaffe i en av kannorna!")
    elif (detection["name"] == "empty coffee"):
        print("Det är troligtvis (", round(detection["percentage_probability"], 2), "%) inget kaffe i en av kannorna.")

sys.stdout.flush()