from paddleocr import PaddleOCR, draw_ocr
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # downloads model if needed
result = ocr.ocr(r'OCR images cropped\1.jpeg', cls=True)

for line in result:
    for word in line:
        print(word[1][0])  # prints recognized text
'''print(result)'''
