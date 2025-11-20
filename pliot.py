import numpy as np
import matplotlib.pyplot as plt

def calculate_I_iter(x):
    """
    计算 I_iter 的值，基于给定的分段函数公式
    """
    y = np.zeros_like(x)
    
    # 第一段: x < 0.1
    mask1 = x < 0.1
    y[mask1] = -25 * x[mask1] + 10
    
    # 第二段: 0.1 <= x < 0.45
    mask2 = (x >= 0.1) & (x < 0.45)
    y[mask2] = 17.14 * (x[mask2]**2) - 15.14 * x[mask2] + 10
    
    # 第三段: x >= 0.45
    mask3 = x >= 0.45
    y[mask3] = 5.66 * (x[mask3]**2) - 2.93 * x[mask3] + 7.2
    
    return y

x = np.linspace(0,1.0,1000)
y = calculate_I_iter(x)
n = np.floor(y) 

plt.figure(figsize=(10,6))

plt.plot(x,y,label=r"$I_{iter}$",color='blue',linewidth=2)#plot:把点连接成线
plt.plot(x,n,label=r"$N_{iter}$",color='red',linestyle='--',alpha=0.7)

plt.axvline(x=0.1,color="gray",linestyle=":",alpha=0.5)#竖直线
plt.axvline(x=0.45,color="gray",linestyle=":",alpha=0.5)

plt.title('number of iter',fontsize=14)
plt.xlabel('Normalized img mean value',fontsize=12)
plt.ylabel('Iterations',fontsize=12)

plt.legend()#自动显示标签

plt.grid(True,which='both',linestyle='--',alpha=0.5)#网格

plt.ylim(0,12)
plt.xlim(0,1)

plt.savefig('Curve.png',dpi=300,bbox_inches='tight')

plt.show()
