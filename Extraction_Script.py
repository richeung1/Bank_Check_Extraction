# Installing dependencies
from imageai.Detection.Custom import CustomObjectDetection
from PIL import Image
import cv2
import os


# # Getting image path
image_path = 'dummycheck1.JPG'


class Extraction:
    def results(self):

        # Creating directory to store updated images
        os.makedirs("detected_objects", exist_ok=True)
        image_result = "detected_objects/" + image_path[:-4] + "_result.JPG"
        image_result_final = image_result[:-4] + "_final.JPG"

        # Detecting 4 classes: check number, date, payee, and amount and saving as a new check image
        detector = CustomObjectDetection()
        detector.setModelTypeAsYOLOv3()
        detector.setModelPath("model\yolov3_Bank_Checks_mAP-0.89818_epoch-79.pt")
        detector.setJsonPath("model\Bank_Checks_yolov3_detection_config.json")
        detector.loadModel()
        detections = detector.detectObjectsFromImage(input_image=image_path,
                                                        output_image_path=image_result, 
                                                        minimum_percentage_probability=90, 
                                                        extract_detected_objects=False)

        # Applying Non-Maximum Suppression to the detected objects in the new image in case there are duplicate objects
        min_probability = 90
        boxes = []
        scores = []
        labels = []
        for detection in detections:
            boxes.append(detection["box_points"])
            scores.append(detection["percentage_probability"])
            labels.append(detection["name"])

        indices = cv2.dnn.NMSBoxes(boxes, scores, min_probability/100, 0.3)

        # Keeping the highest accuracy bounding boxes on the new check image
        image = cv2.imread(image_path)
        for i in indices.flatten():
            x1, y1, x2, y2 = boxes[i]
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(image, labels[i] + " : " + str(scores[i]), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Saving a new image with the new bounding boxes
        cv2.imwrite(image_result_final, image)

        # Extract each object as a JPEG file and putting it in the detected_objects folder created earlier
        for i in indices.flatten():
            label = labels[i]
            score = scores[i]
            x1, y1, x2, y2 = boxes[i]
            object_img = image[y1:y2, x1:x2]
            object_filename = f"detected_objects/{label}_{score:.2f}.jpg"
            cv2.imwrite(object_filename, object_img)