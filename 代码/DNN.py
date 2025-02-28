import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
print(tf.__version__)

# 创建项目子目录
def makepath(Subdirectory):
    try:
        os.mkdir(Subdirectory)
    except FileExistsError:
        pass

# 原始图像文件
raw_filepath_list = [".\\cars", ".\\flowers", ".\\cats", ".\\dogs", ".\\birds",".\\fish",".\\insect",".\\preds"]
# 像素统一化之后的文件
standard_filepath_list = [".\\cars_128", ".\\flowers_128", ".\\cats_128", ".\\dogs_128", ".\\birds_128", ".\\fish_128", ".\\insect_128", ".\\preds_128"]

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
birds_file_list    = get_file_list(filepath='.\\birds',    ext_list=['.jpg'])
fish_file_list    = get_file_list(filepath='.\\fish',    ext_list=['.jpg'])
insect_file_list    = get_file_list(filepath='.\\insect',    ext_list=['.jpg'])
# 待预测未知图像
preds_file_list    = get_file_list(filepath='.\\preds',   ext_list=['.jpg'])

#批量改变图片像素
def imgresize(file_list,savepath, width, height):
    for filepath in file_list:
        print(filepath)
        try:
            im = Image.open(filepath)
            new_im =im.resize((width, height))
            new_im.save(savepath + '\\' + filepath[filepath.rfind('\\')+1:])
        except OSError as e:
            print(e.args)

imgresize(file_list=cars_file_list,    savepath=".\\cars_128",    width=128, height=128)
imgresize(file_list=flowers_file_list, savepath=".\\flowers_128", width=128, height=128)
imgresize(file_list=cats_file_list,    savepath=".\\cats_128",    width=128, height=128)
imgresize(file_list=dogs_file_list,    savepath=".\\dogs_128",    width=128, height=128)
imgresize(file_list=birds_file_list,    savepath=".\\birds_128",    width=128, height=128)
imgresize(file_list=fish_file_list,    savepath=".\\fish_128",    width=128, height=128)
imgresize(file_list=insect_file_list,    savepath=".\\insect_128",    width=128, height=128)

# 待预测未知图像
imgresize(file_list=preds_file_list,   savepath=".\\preds_128",   width=128, height=128)


# 获取像素处理后的图像列表
cars_128_file_list    = get_file_list(filepath='.\\cars_128',    ext_list=['.jpg'])
flowers_128_file_list = get_file_list(filepath='.\\flowers_128', ext_list=['.jpg'])
cats_128_file_list    = get_file_list(filepath='.\\cats_128',    ext_list=['.jpg'])
dogs_128_file_list    = get_file_list(filepath='.\\dogs_128',    ext_list=['.jpg'])
birds_128_file_list    = get_file_list(filepath='.\\birds_128',    ext_list=['.jpg'])
fish_128_file_list    = get_file_list(filepath='.\\fish_128',    ext_list=['.jpg'])
insect_128_file_list    = get_file_list(filepath='.\\insect_128',    ext_list=['.jpg'])
# 待预测未知图像(128x128)
preds_128_file_list   = get_file_list(filepath='.\\preds_128',   ext_list=['.jpg'])

# 所有图像的列表
file_list_all = cars_128_file_list +  flowers_128_file_list + cats_128_file_list + dogs_128_file_list+birds_128_file_list+fish_128_file_list+insect_128_file_list
len(file_list_all)

# 图像转化为像素点数组
Pixel_list = []
for filename in file_list_all:
    im = Image.open(filename)
    im_L = im.convert("L")
    Core = im_L.getdata()
    img_arr = np.array(Core,dtype='float32') / 255.0
    img_list = img_arr.tolist()
    Pixel_list.extend(img_list)

#像素点值转化为数组
X = np.array(Pixel_list).reshape(len(file_list_all), 128, 128)


# 分类标签
class_names = ['car', "flower", "cat", "dog","birds","fish","insect"]
#用字典储存图像信息
dict_label = {0:'car', 1:'flower', 2:'cat', 3:'dog',4:'birds',5:'fish',6:'insect'}

#用列表输入标签，0表示汽车，1表示花,2表示猫,3表示狗,4表示鸟,5表示鱼,6表示虫
label =   [0]*len(cars_128_file_list) \
        + [1]*len(flowers_128_file_list) \
        + [2]*len(cats_128_file_list)\
        + [3]*len(dogs_128_file_list)\
        + [4]*len(birds_128_file_list)\
        + [5]*len(fish_128_file_list)\
        + [6]*len(insect_128_file_list)

y = np.array(label)
#按照4:1的比例将数据划分训练集和测试集
train_images, test_images, train_labels, test_labels = train_test_split(
        X, y, test_size=0.2, random_state=0)

plt.figure()
plt.imshow(train_images[9])
plt.colorbar()
plt.grid(False)
#显示来自训练集的前25个图像，并在每个图像下面显示类名。
#验证数据的格式是否正确，准备构建神经网络
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])

#第一个输入层有128个节点(或神经元)。
#第二个(也是最后一个)层是2个节点的softmax层————返回一个2个概率分数的数组，其和为1。
#每个节点包含一个分数，表示当前图像属于两个类别的概率

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(128, 128)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(len(class_names))
])

# 编译模型
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)


history = model.fit(train_images, train_labels, epochs=50)

epochs=50
# 获取训练过程中的准确率和损失值
train_acc = history.history['accuracy']
# val_acc=history.history['val_accuracy']
train_loss = history.history['loss']
# val_loss=history.history['val_loss']

epochs_range=range(epochs)

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.suptitle("Training Accuracy And Loss")
# 绘制准确率曲线
plt.plot(epochs_range,train_acc,label='Training Accuracy')
# plt.plot(epochs_range,val_acc,label='Validation accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.subplot(1,2,2)

# 绘制损失曲线
plt.plot(epochs_range,train_loss,label='Training Loss')
# plt.plot(epochs_range,val_acc,label='Validation Loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()

# probability_model
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
pred_label = dict_label[np.argmax(predictions[0])]
print(pred_label)


# 定义画图函数
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = '#00BC57'
    else:
        color = 'red'
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(len(class_names)))
    plt.yticks([])
    thisplot = plt.bar(range(len(class_names)), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('#00BC57')


# 让我们看看一张图片，预测标签和真实标签
i = 2
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions[i], test_labels)
plt.show()

# 绘制预测标签和真实标签以及预测概率柱状图
# 正确的预测用绿色表示，错误的预测用红色表示
num_rows = 4
num_cols = 4
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.savefig("./pred80.png", dpi=300)
plt.show()

# 最后，利用训练后的模型对单个图像进行预测。
# 从测试数据集中获取第7个图像
img = test_images[6]
plt.imshow(img, cmap=plt.cm.binary)

# 将图像添加到唯一的成员批处理中.
img = (np.expand_dims(img, 0))
print(img.shape)

# 预测图像:
probability_single = probability_model.predict(img)
print(probability_single)
pred_label = dict_label[np.argmax(probability_single[0])]
print(pred_label)
# 可视化预测概率
plot_value_array(1, probability_single[0], test_labels)
_ = plt.xticks(range(len(class_names)), class_names, rotation=45)

# 对于一个图像
im = Image.open(preds_128_file_list[0])
width, height = im.size
im_L = im.convert("L")
Core = im_L.getdata()
arr1 = np.array(Core, dtype='float32') / 255.0
list_img = arr1.tolist()
img = np.array(list_img).reshape(width, height)
pred_labels = np.array([0])
print(img.shape)
plt.imshow(img, cmap=plt.cm.binary)
# 将图像添加到唯一成员的批处理文件中.
img = (np.expand_dims(img, 0))
print(img.shape)

# 预测图像概率:
probability_single = probability_model.predict(img)
print(probability_single)
# 可视化概率
plot_value_array(0, probability_single[0], pred_labels)
_ = plt.xticks(range(len(class_names)), class_names, rotation=45)

# 可视化预测结果
np.argmax(probability_single[0])
pred_label = dict_label[np.argmax(probability_single[0])]


def img2label(filename):
    im = Image.open(filename)
    width, height = im.size
    im_L = im.convert("L")
    Core = im_L.getdata()
    arr1 = np.array(Core, dtype='float32') / 255.0
    list_img = arr1.tolist()
    img = np.array(list_img).reshape(width, height)
    img = (np.expand_dims(img, 0))
    predictions_single = model.predict(img)
    return np.argmax(predictions_single[0])


# 得到多个图像的分类字典编号
pred_labels = [img2label(filename) for filename in preds_128_file_list]

# 字典编号翻译成对应的标签
for i, num in enumerate(pred_labels):
    print('第' + str(i + 1) + '张图像识别为: ' + dict_label[num])

# 可视化多个图像
M = []
for filename in preds_128_file_list:
    im = Image.open(filename)
    width, height = im.size
    im_L = im.convert("L")
    Core = im_L.getdata()
    arr1 = np.array(Core, dtype='float') / 255.0
    list_img = arr1.tolist()
    M.extend(list_img)

pred_images = np.array(M).reshape(len(preds_128_file_list), width, height)
num_rows = 2
num_cols = 4
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i, img in enumerate(pred_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(img)
    plt.xlabel(class_names[pred_labels[i]])
