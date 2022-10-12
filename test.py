from mmocr.ocr import MMOCR

ocr = MMOCR(recog='CRNN', det='DB_r18')
ocr.readtext('demo/demo_text_ocr.jpg', show=True)
