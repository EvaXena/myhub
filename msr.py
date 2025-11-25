import cv2
import numpy as np
from memory_profiler import profile
import os
import matplotlib.pyplot as plt
#文件读取
# for root,dirs,files in os.walk(path):
#     for file in files:
#         if file.endswith(('.jpg','.png'))

#使用装饰器:pip install line_profiler,memory_profiler
#如看内存：from memory_profiler import profile
#在需要的函数上加装饰器@profile
#终端运行脚本  kernprof -l -v xxx.py
#-l:逐行模式  -v:直接在终端显示结果
#可视化分析：pip install snakeviz
#终端运行脚本  python -m cProfile -o output.prof xxx.py

#查看.lprof: python -m line_profiler xxx.py.lprof
#   输出到txt里慢慢看：python -m line_profiler xxx.py.lprof > result.txt

##==============================================================================
##PD
SIGMA_S = 1000
SIGMA_M = 1000
SIGMA_L = 1000

W1 = 1/3
W2 = 1/3
W3 = 1/3 

ROOT_PATH = 'input/'
##==============================================================================


@profile
def ssr_cal(img,sigma):
    retinex = np.log10(img) - np.log10(cv2.GaussianBlur(img,(0,0),sigma))
    return retinex

@profile
def msr_cal(img,sigma_list,w_list):
    msr = w_list[0] * ssr_cal(img,sigma_list[0]) + w_list[1] * ssr_cal(img,sigma_list[1]) + w_list[2] * ssr_cal(img,sigma_list[2])
    return msr


def retinex_process():
    sigma_list = [SIGMA_S,SIGMA_M,SIGMA_L]
    w_list = [W1,W2,W3]
    for root,dirs,files in os.walk(ROOT_PATH):
        for file in files:
            if file.endswith(('.jpg','.png')):
                fullpath = os.path.join(root,file)
                img = cv2.imread(fullpath)
                img = np.float64(img)
                retinex = msr_cal(img,sigma_list,w_list)
                for i in range(retinex.shape[2]):
                    # unique,counts = np.unique(retinex[:,:,i],return_counts=True)
                    min = np.percentile(retinex[:,:,i],1)
                    max = np.percentile(retinex[:,:,i],99)

                    retinex[:,:,i] = (retinex[:,:,i] - min) / (max - min) * 255

                retinex = np.uint8(np.clip(retinex,0,255))
                cv2.imwrite(f"{file}.png",retinex)
    return 0

if __name__ == "__main__":
    pic = retinex_process()
    


