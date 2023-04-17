## DACL-LSBG目标检测模型在pytorch当中的实现
---Deep Learning-based Accurate Center Localization for Low Surface Brightness Galaxies


### 所需环境
scipy==1.2.1
numpy==1.17.0
matplotlib==3.1.2
opencv_python==4.1.2.30
torch==1.2.0
torchvision==0.4.0
tqdm==4.60.0
Pillow==8.2.0
h5py==2.10.0

在pycharm终端输入 pip install -r requirements.txt 批量安装依赖包
 
### 数据集
LSBG数据集存储在在Galaxy_data/Galaxy_1文件夹中，包括图像和标签。

### 训练 
使用train.py的默认参数用于训练LSBG数据集，直接运行train.py即可开始训练。
**训练的权重默认保存在logs文件夹中，可以在train.py中修改。**


### 预测
训练结果预测需要用到两个文件，分别是centernet.py和predict.py。  
首先需要去centernet.py里面修改model_path。
   **model_path指向训练好的权值文件，在logs文件夹里。**   
然后运行predict.py进行检测。
   **默认对13_val.txt中的图像进行检测，检测结果保存在map_out文件夹中。
可以修改predict.py中的predict_path和map_out_path改变检测的图像和结果输出的文件夹**



## Reference
https://github.com/liangzengxu/DACL-LSBG
