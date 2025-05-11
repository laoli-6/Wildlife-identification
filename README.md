![image](https://github.com/user-attachments/assets/ac50c8b4-912c-44be-a4d2-39f24b817d14)#   深度学习的视频中动物识别的系统介绍

## 1、模型所需数据结构介绍

YOLO和CNN的训练数据的目录格式如下所示

YOLO：

![image](https://github.com/user-attachments/assets/61b72f5f-6713-419e-bb9e-4525a463fe5e)

在images和labels中需要重命名相同，如果不相同会导致照片的识别框消失，训练过程可能会报错。

针对数据配置文件，需要git下yolo的仓库，在仓库有个配置文件需要进行更改。

使用我的数据集不需要进行数据配置文件的修改，以下是数据集获取网站

YOLO数据集:https://pan.baidu.com/s/1UQ69LVaPxa0grDW9HBn7hg?pwd=g9ib 提取码: g9ib

CNN：

![image](https://github.com/user-attachments/assets/c83f76d7-f55e-48a0-9995-c59a244083d7)

CNN的分配很简单，只需要把照片放在相对应的文件里并设置好文件夹重命名操作，数据获取如下

CNN数据集:https://pan.baidu.com/s/1HV22FjnNzR0I9gttYemLfw?pwd=frss 提取码: frss

## 2、文件目录列表格式介绍

![image](https://github.com/user-attachments/assets/c6f1224d-64ac-4cbd-b8bb-1f459fed14fc)


.idea文件夹主要是自动生成的项目配置文件夹，不用管

YOLO_model是保存模型的文件，需要保存新的模型存储到该文件夹

static是上传的图像/视频和经过模型处理过的图像/视频

templates是存放index.html也就是主界面

app.py是主程序运行的文件

## 3、本地部署及相对应的库安装

想进行本地部署可以直接进行git操作或者采用Download zip进行下载，进行解压到桌面即可

解压完成之后文件目录格式如上所示

![image](https://github.com/user-attachments/assets/c9533835-3118-4b55-81c3-b8b556387cab)


点击右键通过pycharm打开这个项目，点开app.py没有安装库可能会出现报错提示，看报错哪一个安装哪一个

必须安装以下的库，就算打开不报错也要安装，是YOLO的配置文件

```
pip install ultralytics
```

解决完报错问题，就可以运行主程序，完成动物的识别

如果还是有问题可以发邮箱进行询问bravo202109@163.com

## 4、第一次写这个东西，可能有的东西不是很合适，希望多多见谅，也可以指出点意见，感谢各位大神。
