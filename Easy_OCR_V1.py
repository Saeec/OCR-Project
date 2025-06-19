import cv2
import easyocr
import numpy as np
import supervision as sv
import os
import time
import csv

class EasyOCRBatchProcessor:
    def __init__(self, image_directory, output_directory, gpu=True):
        self.image_directory = image_directory
        self.output_directory = output_directory
        self.reader = easyocr.Reader(['en'], gpu=gpu,download_enabled=False,model_storage_directory=r'models')

        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
        self.summary_lines = []
        self.csv_rows = []

    def rotate_image(self, img, angle):
        if angle == 0:
            return img
        elif angle == 90:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(img, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            raise ValueError("Angle must be 0, 90, 180, or 270 degrees.")

    def process_image(self, image_path, skip_rotation=False):
        start_time = time.time()

        image = cv2.imread(image_path)
        if image is None:
            print(f"Image not found or cannot be read: {image_path}")
            return

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if skip_rotation:
            best_image = gray.copy()
            best_result = self.reader.readtext(best_image)
            best_angle = "manual"
        else:
            best_angle = 0
            best_score = float('-inf')
            best_result = []
            best_image = gray.copy()

            for angle in [0, 90, 180, 270]:
                rotated_img = self.rotate_image(gray, angle)
                result = self.reader.readtext(rotated_img)
                total_conf = sum([res[2] for res in result]) if result else 0
                if total_conf > best_score:
                    best_score = total_conf
                    best_angle = angle
                    best_result = result
                    best_image = rotated_img.copy()

        print(f"{os.path.basename(image_path)}: Rotation used: {best_angle}°")

        best_image = cv2.cvtColor(best_image, cv2.COLOR_GRAY2BGR)

        xyxy, confidences, class_ids, labels = [], [], [], []
        image_summary = []
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        if not best_result:
            print(f"No text detected in {os.path.basename(image_path)}.")
            annotated_image = best_image
            self.summary_lines.append(f"Image: {image_name}\n    No text detected.\n")
            self.csv_rows.append([image_name, "", "", "", "", "", ""])
        else:
            for detection in best_result:
                bbox, text, confidence = detection[0], detection[1], detection[2]

                # Replace text containing "shaw" (case-insensitive) by "RENISHAW"
                if "shaw" in text.lower():
                    text = "RENISHAW"

                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                x_min = int(min(x_coords))
                y_min = int(min(y_coords))
                x_max = int(max(x_coords))
                y_max = int(max(y_coords))
                w = x_max - x_min
                h = y_max - y_min
                xyxy.append([x_min, y_min, x_max, y_max])
                confidences.append(float(confidence))
                class_ids.append(0)
                labels.append(str(text))
                image_summary.append(
                    f"    Detected Text: {text}\n"
                    f"    Coordinates: ({x_min}, {y_min}, {w}, {h})\n"
                    f"    Confidence: {confidence:.4f}\n"
                )
                self.csv_rows.append([
                    image_name,
                    text,
                    x_min,
                    y_min,
                    w,
                    h,
                    f"{confidence:.4f}"
                ])

            detections = sv.Detections(
                xyxy=np.array(xyxy, dtype=int),
                confidence=np.array(confidences, dtype=float),
                class_id=np.array(class_ids, dtype=int)
            )
            box_annotator = sv.BoxAnnotator()
            label_annotator = sv.LabelAnnotator()
            annotated_image = box_annotator.annotate(scene=best_image, detections=detections)
            annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
            self.summary_lines.append(f"Image: {image_name}\n" + "".join(image_summary))

        output_path = os.path.join(self.output_directory, f"{image_name}_output.png")
        cv2.imwrite(output_path, annotated_image)
        elapsed_time = time.time() - start_time
        print(f"Annotated image saved as {output_path}")
        print(f"Execution time for {os.path.basename(image_path)}: {elapsed_time:.2f} seconds")

    def process_all_images(self):
        supported_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        for filename in os.listdir(self.image_directory):
            if filename.lower().endswith(supported_exts):
                image_path = os.path.join(self.image_directory, filename)
                self.process_image(image_path)
        # Write summary text file
        summary_path = os.path.join(self.output_directory, "ocr_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.summary_lines))
        print(f"Summary file saved as {summary_path}")
        # Write CSV file
        csv_path = os.path.join(self.output_directory, "ocr_summary.csv")
        with open(csv_path, "w", newline='', encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Image Name", "Detected Text", "X", "Y", "W", "H", "Confidence"])
            writer.writerows(self.csv_rows)
        print(f"CSV summary file saved as {csv_path}")


if __name__ == "__main__":
    image_directory = r"OCR images cropped"
    output_directory = r"OCR Results"
    processor = EasyOCRBatchProcessor(image_directory, output_directory)

    
    
    # Then process the batch folder
    processor.process_all_images()