

## EMSNet:Efficient Multimodal Symmetric Network for Semantic Segmentation from Remote Sensing Imagery

Optical and DSM, Optical and SAR for segmentation

<br />

 本篇README.md面向EMSNet使用者

## Datasets
├─Train set <br />
&nbsp;&nbsp; ├─dsm(sar) <br />
&nbsp;&nbsp;  ├─opt <br />
&nbsp;&nbsp;  ├─lbl <br />
├─Val set <br />
&nbsp;&nbsp;  ├─dsm(sar) <br />
&nbsp;&nbsp;  ├─opt <br />
&nbsp;&nbsp;  ├─lbl <br />
├─Test set <br />
&nbsp;&nbsp;  ├─dsm(sar) <br />
&nbsp;&nbsp;  ├─opt <br />
&nbsp;&nbsp;  ├─lbl <br />

## Getting Started
#### Train
    python train.py --train_data_root train_path --val_data_root val_path --model_path weight_path --base_lr 0.001 --batch_size 16
#### Predict
    python predict.py --test_data_root test_path --model_path weight_path 
不同数据集需要修改脚本中的类别数量和是否统计背景
## Requirments

* python, cv2, numpy, PIL
* pytorch, tensorboardX






