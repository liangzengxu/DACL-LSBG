## Implementation of GalCenterNet Model in PyTorch
---Automatic Search for Low Surface Brightness Galaxies from SDSS images Using Deep Learning

### Prerequisites
scipy==1.2.1
numpy==1.17.0
matplotlib==3.1.2
opencv_python==4.1.2.30
torch==1.2.0
torchvision==0.4.0
tqdm==4.60.0
Pillow==8.2.0
h5py==2.10.0

Install all dependencies in bulk by entering `pip install -r requirements.txt` in the PyCharm terminal.

### Dataset
The LSBG dataset is stored in the `Galaxy_data/Galaxy_1` directory, which includes both images and labels.

### Training 
The default parameters in `train.py` are used for training the LSBG dataset. Run `train.py` directly to start the training.
**Training weights are saved in the `logs` directory by default, but this can be modified within `train.py`.**

### Prediction
Prediction of the training results requires two files: `centernet.py` and `predict.py`.
First, you need to modify the `model_path` inside `centernet.py`.
   **`model_path` should point to the trained weights file located in the `logs` directory.**  
Then, run `predict.py` to perform the detection.
   **By default, it detects images in `13_val.txt`. The detection results are saved in the `map_out` directory. You can modify the `predict_path` and `map_out_path` inside `predict.py` to change the images to be detected and the output directory for the results.**

## Reference
https://github.com/liangzengxu/DACL-LSBG
