

## EMSNet:Efficient Multimodal Symmetric Network for Semantic Segmentation from Remote Sensing Imagery

Optical and DSM, Optical and SAR for segmentation

<br />

 本篇README.md面向EMSNet推理使用者
 
## Models
| Dataset     | OA          | mIoU          | Kappa          | Download(Baidu Drive)                                           |
| :----:      |    :----:   |       :----:  | :----:         | :----:                                                          |
| Potsdam     | 93.1        | 82.7          | 90.4           | [Model](https://pan.baidu.com/s/10ZOnvlccKk0Wk7eqFUjHPQ?pwd=ied2)        |
| Vaihingen   | 94.9        | 70.2          | 87.4           | [Model](https://pan.baidu.com/s/10ZOnvlccKk0Wk7eqFUjHPQ?pwd=ied2)        |
| WHU-OPT-SAR | 78.4        | 46.2          | 69.7           | [Model](https://pan.baidu.com/s/10ZOnvlccKk0Wk7eqFUjHPQ?pwd=ied2)        |
| WHU_kd      | 81.2        | 49.1          | 73.3           | [Model](https://pan.baidu.com/s/10ZOnvlccKk0Wk7eqFUjHPQ?pwd=ied2)        |
| DFC2023     | 89.6        | 74.2          | 69.1           | [Model](https://pan.baidu.com/s/10ZOnvlccKk0Wk7eqFUjHPQ?pwd=ied2)        |
| DFC2023_kd  | 90.7        | 77.1          | 73.2           | [Model](https://pan.baidu.com/s/10ZOnvlccKk0Wk7eqFUjHPQ?pwd=ied2)        |

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

    python predict_xx.py --test_data_root dataset_path --model_path weight_path 

## Requirments

* python, cv2, numpy, PIL
* pytorch






