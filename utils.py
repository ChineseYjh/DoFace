import os
import sys
import subprocess
import time
import numpy as np
from PIL import Image
from .facetools import parsing as facetools_parsing


"""
====================================CONSTANT====================================
"""

part2label = {'background':0,
              'skin':1, 
              'l_brow':2, 
              'r_brow':3, 
              'l_eye':4, 
              'r_eye':5, 
              'eye_g':6, 
              'l_ear':7, 
              'r_ear':8, 
              'ear_r':9,
              'nose':10, 
              'mouth':11, 
              'u_lip':12, 
              'l_lip':13, 
              'neck':14, 
              'neck_l':15, 
              'cloth':16, 
              'hair':17, 
              'hat':18
             }


meitong_l=['shenlanse','yamajin']
chuncai_l=['nanguase','niuxuese','yinghuafen']

"""
====================================FUNCTION====================================
"""


def crop_part(src_path,dst_path,parts,cp='./facetools/parsing/cp/cp_parsing.pth'):
    """
    @params
        src_path: str
        dst_path: str
        parts: list(str)
    @return
        ret: Image, tuple([int],[int]),int,int,Image
    """
    img=Image.open(src_path)
    labels=[part2label[part] for part in parts]
    box,index,box_up,box_le=crop_part_(img,labels,cp=cp)
    if box==None:
        return None,None,None,None,None
    box.save(dst_path,quality=95)
    return img,index,box_up,box_le,box


def recover_part(src_img,ori_box,index,part_path,dst_path,box_up,box_le,save=False):
    """
    @params:
        src_img: Image
        ori_box: Image
        index: tuple(list(int),list(int))
        part_path: str
        dst_path: str
        box_up: int
        box_le: int
    @return:
        dst: Image
    """
    src_np=np.array(src_img).astype(np.uint8)
    box_np=np.array(Image.open(part_path).resize((ori_box.width,ori_box.height),Image.ANTIALIAS)).astype(np.uint8)
    src_np[index]=box_np[(index[0]-box_up,index[1]-box_le)]
    dst=Image.fromarray(src_np)
    if save==True:
        dst.save(dst_path,quality=95)
    return dst

def crop_part_(img,labels,cp):
    """
    @params
        img: Image
        labels: list(int)
    @return:
        ret: Image,tuple([int],[int]),int,int
    """
    def check(mask,labels):
        ret=np.zeros_like(mask).astype(np.bool)
        for label in labels:
            ret=ret+(mask==label)
        return ret.astype(np.bool)
    img_np=np.array(img).astype(np.uint8)
    mask=facetools_parsing(img,cp=cp).astype(np.uint8)
    index=np.where(check(mask,labels))
    if(len(index[0])==0):
        return None,None,None,None
    box_up=np.min(index[0])
    box_lo=np.max(index[0])
    box_le=np.min(index[1])
    box_ri=np.max(index[1])
    box=img_np[box_up:box_lo+1,box_le:box_ri+1,:]
    box=Image.fromarray(box)
    return box,index,box_up,box_le


def set_filename(path='./dataset'):
    """
    regularize the file names of dataset
    """
    for par,dirs,files in os.walk(path):
        if(len(files)>0):
            for num,file in enumerate(files):
                os.rename(os.path.join(par,file),os.path.join(par,f'{num}.png'))
                

def style_transfer(path_name,style,gpu_ids='1'):
    """
    @params:
        path_name: str
        style: str
    @return:
        p: int
    """
    cmd1='cd ./DoFace/pytorchCycleGANandPix2Pix/'
    cmd2=f'python test.py --dataroot ../../tmp/{path_name}/testA --results_dir ../../tmp/{path_name}/result --name raw2{style} --model test --no_dropout --gpu_ids {gpu_ids}'
    cmd=cmd1+' && '+cmd2
    p=subprocess.call(cmd,shell=True)
    return p


def chuncai(img_path,style,dst_path,save=False,cp='./facetools/parsing/cp/cp_parsing.pth',gpu_ids='1'):
    """
    @params:
        img_path: str
        style: str
        dst_path: str
    @return:
        ret: Image
    @func:
        This suits for images including only one face
    """
    
    print('prepare')
    
    file_name=time.strftime('%Y%m%d%H%M%S',time.localtime(time.time()))+f'-{style}'
    parts=['u_lip','l_lip']
    box_path=f'./tmp/{file_name}/testA/'
    if not os.path.exists(box_path):
        os.makedirs(box_path)
        
    print('crop')    
    
    img,index,box_up,box_le,box=crop_part(img_path,os.path.join(box_path,'part.png'),parts,cp=cp)
    if img==None:
        print("Can't detect mouth!")
        return None
    
    print('transfer')
    
    state=style_transfer(file_name,style,gpu_ids)
    if state!=0:
        return None
    
    print('recover')
    
    ret=recover_part(img,box,index,f'./tmp/{file_name}/result/raw2{style}/test_latest/images/part_fake.png',
                     dst_path,box_up,box_le,save=save)
    return ret


def meitong(img_path,style,dst_path,save=False,cp='./facetools/parsing/cp/cp_parsing.pth',gpu_ids='1'):
    """
    @params:
        img_path: str
        style: str
        dst_path: str
    @return:
        ret: Image
    @func:
        This suits for images including only one face
    """
    
    print("prepare")
    
    file_name=time.strftime('%Y%m%d%H%M%S',time.localtime(time.time()))+f'-{style}'
    parts_l=[['l_eye'],['r_eye']]
    box_le_path=f'./tmp/{file_name}-left/testA/'
    box_ri_path=f'./tmp/{file_name}-right/testA/'
    if not os.path.exists(box_le_path):
        os.makedirs(box_le_path)
    if not os.path.exists(box_ri_path):
        os.makedirs(box_ri_path)
    
    print("left\ncrop left")
    
    img,index_le,box_up_le,box_le_le,box_le=crop_part(img_path,os.path.join(box_le_path,'part.png'),parts_l[0],cp=cp)
    mid=Image.open(img_path)
    if img!=None:
        
        print("transfer left")
        
        state=style_transfer(file_name+'-left',style+'-left',gpu_ids)
        if state==0:
            
            print("recover left")
            
            mid=recover_part(img,box_le,index_le,
                             f'./tmp/{file_name}-left/result/raw2{style}-left/test_latest/images/part_fake.png',
                             dst_path,box_up_le,box_le_le,save=False)
        
        
    print("right\ncrop right")
    
    img,index_ri,box_up_ri,box_le_ri,box_ri=crop_part(img_path,os.path.join(box_ri_path,'part.png'),parts_l[1],cp=cp)
    if img!=None:
        
        print("transfer right")
        
        state=style_transfer(file_name+'-right',style+'-right',gpu_ids)
        if state==0:
            
            print("recover right")
            
            ret=recover_part(mid,box_ri,index_ri,
                             f'./tmp/{file_name}-right/result/raw2{style}-right/test_latest/images/part_fake.png',
                             dst_path,box_up_ri,box_le_ri,save=save)
    if index_ri==index_le==None:
        return None
    return ret
    
def doface(img_path,style,dst_path,save=False,cp='./facetools/parsing/cp/cp_parsing.pth',gpu_ids='1'):
    """
    @params:
        img_path: str
        style: str
        dst_path: str
    @return:
        ret: Image
    @func:
        This suits for images including only one face
        This function supports :
        (1)chuncai(唇彩)
            niuxuese：牛血色
            nanguase：南瓜色
            yinghuafen：樱花粉
        (2)meitong(美瞳)
            shenlanse：深蓝色
            yamajin：亚麻金
    """
    ret=None
    start_time=time.time()
    if style in meitong_l:
        ret=meitong(img_path,style,dst_path,save=save,cp=cp,gpu_ids=gpu_ids)
    elif style in chuncai_l:
        ret=chuncai(img_path,style,dst_path,save=save,cp=cp,gpu_ids=gpu_ids)
        
    print(f"Total time cost: {time.time()-start_time} s.")
    return ret