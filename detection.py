import torch
import cv2
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
def detect_objects(image_path, output_path='output.jpg'):
    image = cv2.imread(r"C:\Users\Ramya Deepthi\OneDrive\Desktop\python-Yolovo5 project\car1.jpg")

    if image is None:
        print ("Could not load the image. Check the file name.")
        return
    results = model(image)
    detections = results.pandas().xyxy[0]
    if detections.empty:
        print("No objects detected in the image.")
        return
    for index, row in detections.iterrows():
        x1, y1 = int(row['xmin']), int(row['ymin'])
        x2, y2 = int(row['xmax']), int(row['ymax'])
        label = f"{row['name']} ({row['confidence']:.2f})"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imwrite(output_path, image)
    print(f"Detection complete. Output saved as '{output_path}'.")
detect_objects("car1.jpg")