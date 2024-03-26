from ultralytics import YOLO
from os.path import join
from os import listdir

model = YOLO(join("C:/Users/zincuonn/Documents/Python/model_test/model", "3_2/best.pt"))

for i in listdir("C:/Users/zincuonn/Documents/Python/model_test/test"):
    results = model.predict(join("C:/Users/zincuonn/Documents/Python/model_test/test", i), save=True, conf=.1)
#model.predict(source=0, conf=.3, show=True)

for result in results:
    print(result.boxes, result.probs)

    