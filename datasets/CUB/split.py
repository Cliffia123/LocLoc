import os
bounding_boxes = '/GPUFS/nsccgz_ywang_zfd/caoxz/transformer/LCTR-master/datasets/CUB/bounding_boxes.txt'
test = '/GPUFS/nsccgz_ywang_zfd/caoxz/transformer/LCTR-master/datasets/CUB/test.txt'
image_id_list = []
with open(test) as f:
    for line in f:
        info = line.strip().split()
        image_id = info[0]
        image_id_list.append(image_id)

image_id_list_box = []
with open(bounding_boxes) as f:
    for line in f:
        info = line.strip().split()
        image_id_list_box.append(line)

f = open('test_boxes.txt', 'a')
f2 = open('train_boxes.txt', 'a')

for line in image_id_list_box:
    info = line.strip(). vå

    if info[0] in image_id_list:
        f.writelines([line])
        print(info)
    # else:
    #     f2.writelines([line])

