import os
import cv2
import time
import csv
from paddleocr import PaddleOCR, draw_ocr
import numpy as np

# Initialize PaddleOCR
ocr = PaddleOCR(ocr = PaddleOCR(
    rec_model_dir="paddle_models\en_PP-OCRv3_rec_infer",
    det_model_dir="paddle_models/en_PP-OCRv3_det_infer",
    cls_model_dir="paddle_models\ch_ppocr_mobile_v2.0_cls_infer",
    use_angle_cls=True,
    use_gpu=False,
    lang='en'
))

# Define folders
input_folder = "OCR images cropped"
output_folder = "OCR Paddle Results"
csv_path = os.path.join(output_folder, "paddlesummary.csv")

# Create output folder
os.makedirs(output_folder, exist_ok=True)

# Supported image extensions
image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

# Prepare CSV file
csv_rows = [["Image Name", "Detected Text", "Box Coordinates", "Confidence"]]

# Process all images
for filename in os.listdir(input_folder):
    if not filename.lower().endswith(image_extensions):
        continue

    image_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_output.jpg")

    print(f"\n🔍 Processing: {filename}")
    start_time = time.time()

    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Error reading: {filename}")
        continue

    result = ocr.ocr(image, cls=True)

    for line in result:
        boxes = [res[0] for res in line]
        texts = [res[1][0] for res in line]
        scores = [res[1][1] for res in line]

        for box, text, score in zip(boxes, texts, scores):
            box_str = str(box)
            csv_rows.append([filename, text, box_str, f"{score:.4f}"])

        # Draw only the text (without confidence)
        image = draw_ocr(image, boxes, texts, font_path='C:/Windows/Fonts/arial.ttf')

    # Save annotated image
    cv2.imwrite(output_path, image)
    elapsed = time.time() - start_time
    print(f"✅ Saved to: {output_path} | Time: {elapsed:.2f}s")

# Write to CSV
with open(csv_path, "w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(csv_rows)

print(f"\n📄 CSV summary saved as: {csv_path}")
