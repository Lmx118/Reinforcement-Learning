import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']

# 创建项目子目录
def makepath(Subdirectory):
    try:
        os.mkdir(Subdirectory)
    except FileExistsError:
        pass

# 原始图像文件
raw_filepath_list = [".\\cars", ".\\flowers", ".\\cats", ".\\dogs", ".\\preds"]
# 像素统一化之后的文件
standard_filepath_list = [".\\cars_128", ".\\flowers_128", ".\\cats_128", ".\\dogs_128", ".\\preds_128"]

for raw_filepath in raw_filepath_list:
    makepath(raw_filepath)
for standard_filepath in standard_filepath_list:
    makepath(standard_filepath)


# 获取每个类别的图片文件列表
def get_file_list(filepath, ext_list):
    file_list = []
    for root, dirs, files in os.walk(filepath):
        for file in files:
            if os.path.splitext(file)[1] in ext_list:
                # file_list.append(os.path.join(root, file))
                file_list.append(os.path.join(filepath, file))
        return file_list


cars_file_list    = get_file_list(filepath='.\\cars',    ext_list=['.jpg'])
flowers_file_list = get_file_list(filepath='.\\flowers', ext_list=['.jpg'])
cats_file_list    = get_file_list(filepath='.\\cats',    ext_list=['.jpg'])
dogs_file_list    = get_file_list(filepath='.\\dogs',    ext_list=['.jpg'])
# 待预测未知图像
preds_file_list    = get_file_list(filepath='.\\preds',   ext_list=['.jpg'])

# img = Image.open(cats_list[0])
# plt.imshow(img)
# plt.axis('off')
# plt.show()

#批量改变图片像素
def imgresize(file_list,savepath, width, height):
    for filepath in file_list:
        print(filepath)
        try:
            im = Image.open(filepath)
            new_im =im.resize((width, height))
            new_im.save(savepath + '\\' + filepath[filepath.rfind('\\')+1:])
            print('图片' + filepath[filepath.rfind('\\')+1:] + '像素转换完成')
        except OSError as e:
            print(e.args)

imgresize(file_list=cars_file_list,    savepath=".\\cars_128",    width=128, height=128)
imgresize(file_list=flowers_file_list, savepath=".\\flowers_128", width=128, height=128)
imgresize(file_list=cats_file_list,    savepath=".\\cats_128",    width=128, height=128)
imgresize(file_list=dogs_file_list,    savepath=".\\dogs_128",    width=128, height=128)
# 待预测未知图像
imgresize(file_list=preds_file_list,   savepath=".\\preds_128",   width=128, height=128)


# 获取像素处理后的图像列表
cars_128_file_list    = get_file_list(filepath='.\\cars_128',    ext_list=['.jpg'])
flowers_128_file_list = get_file_list(filepath='.\\flowers_128', ext_list=['.jpg'])
cats_128_file_list    = get_file_list(filepath='.\\cats_128',    ext_list=['.jpg'])
dogs_128_file_list    = get_file_list(filepath='.\\dogs_128',    ext_list=['.jpg'])
# 待预测未知图像(128x128)
preds_128_file_list   = get_file_list(filepath='.\\preds_128',   ext_list=['.jpg'])

# 所有图像的列表
file_list_all = cars_128_file_list +  flowers_128_file_list + cats_128_file_list + dogs_128_file_list
len(file_list_all)