from roboflow import Roboflow

rf = Roboflow(api_key="") #roboflow api keyiniz
project = rf.workspace("yolo-1aviq").project("pipe-effect") #projeniz
version = project.version(2)
dataset = version.download("yolov8") #kullanılan modeliniz
