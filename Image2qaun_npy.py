import numpy as np
from PIL import Image
import os
from scipy import fftpack
import find_bpp
import matplotlib.pyplot as plt
from math import sqrt
def convert(a, b, path, file_name):
    global out_line
    output, bpp = find_bpp.convert(a,b, path+'//'+file_name)
    np.save('./save/'+ file_name+'-'+str(bpp), output)
    out_line[int(round(bpp,1) * 10)] += 1
out_line = [0] * 250
if __name__ == '__main__':
    path = r'C:\Users\Jin\untitled\valid_32x32\valid_32x32'
    files = os.listdir(path)
    bd_list = [(3,3),(3.5,3.5),(4,4),(4.3,4.3),(2.5,2.5),(2,2)]
    for a,b in bd_list: 
        for i, file in enumerate(files):
            convert(a ,b, path, file)
            if i > 100:
                    break
        
    #plot bpp by range
    x = np.arange(250)
    x_bar = [int(i*0.1) if i % 10 == 0 else '' for i in range(0, 250)]
    plt.bar(x, out_line)
    plt.xticks(x, x_bar)
    plt.show()
