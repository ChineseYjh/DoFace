# DoFace
A package that supports Virtual Makeup function, including **lip color transfer** and **cosmetic contact lenses**. This repository is based on PyTorch.

![唇彩](https://github.com/ChineseYjh/DoFace/blob/master/imgs/%E5%94%87%E5%BD%A9.jpg)
![美瞳](https://github.com/ChineseYjh/DoFace/blob/master/imgs/%E7%BE%8E%E7%9E%B3.jpg)


## Get started
Clone the repository and configurate the environment.
```
git clone git@github.com:ChineseYjh/DoFace.git
pip install -r requirements.txt
```
Use DoFace in your projects.
```
Your project
│   README.md
│   ...
│   foo.py
│
└───DoFace
│
└───directory1
│   
└───...
```
Write codes in `foo.py` like:

```python
import DoFace
img_path='./imgs/test.jpg' # original face image path
style="niuxuese" # it supports "niuxuese", "nanguase" and "yinghuafen" for chuncai(唇彩) ,and "shenlanse", "yamajin" for meitong(美瞳). All styles are supported by doface function.
dst_path='./imgs/res-niuxuese.jpg' # makeup result image path
save=True # whether to save img to dst_path
img=DoFace.doface(img_path=img_path,style=style,dst_path=dst_path,save=save,
                  cp='./DoFace/facetools/parsing/cp/cp_parsing.pth') # cp points to the face-parsing weight path
```
This package now supports:
- chuncai(唇彩)
    - nanguase(南瓜色)
    - niuxuese(牛血色)
    - yinghuafen(樱花粉)
- meitong(美瞳)
    - shenlanse(深蓝色)
    - yamajin(亚麻金)
    
Choose one and assign it to `style` in `foo.py`.


## Acknowledgement
This repository is based on [facetools](https://github.com/zllrunning/facetools) and [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

