import os
from mmocr.ocr import MMOCR

# Models are stored at /home/xiran/.cache/torch/hub/checkpoints
# ocr = MMOCR(recog='CRNN', det='DB_r18')
ocr = MMOCR(recog='ABINet', det='DB_r18')
# ocr = MMOCR(recog='SATRN', det='DB_r50')
# ocr = MMOCR(det='DB_r50')
# ocr = MMOCR(det='DB_r18')
# ocr = MMOCR(det='FCE_IC15')
# ocr = MMOCR(det='FCE_IC15', recog='SATRN')

# input_path = 'demo/map_text/gt'
input_path = 'demo/map_text/HRbicx2'
# input_path = 'demo/map_text/HRbicx3'
# input_path = 'demo/map_text/LRbicx2'
# input_paths = ['demo/map_text/LRbicx3']

# input_paths = ['demo/map_text/' + s for s in ['gt', 'HRbicx2', 'HRbicx3', 'LRbicx2', 'LRbicx3']]


# for input_path in input_paths:
#     out_path = 'demo/out_map_text/' + input_path.split('/')[-1]
#     pic_name = os.listdir(input_path)
#     for name in pic_name:
#         ocr.readtext(os.path.join(input_path, name), img_out_dir=out_path)

pic_name = os.listdir(input_path)
out_path = 'demo/out_map_text/test'
for name in pic_name:
    ocr.readtext(os.path.join(input_path, name), img_out_dir=out_path)
