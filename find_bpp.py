from PIL import Image
import numpy as np
from scipy import fftpack
import huffmanEncode
from bitstream import BitStream
import os
import matplotlib.pyplot as plt
import random
from scipy.special import beta, gamma

# quaility factor = 50
def find_quan(k):
    # Y component
    if k == 0:
        return np.array([[16,  11,  10,  16,  24,  40,  51,  61],
                         [12,  12,  14,  19,  26,  58,  60,  55],
                         [14,  13,  16,  24,  40,  57,  69,  56],
                         [14,  17,  22,  29,  51,  87,  80,  62],
                         [18,  22,  37,  56,  68, 109, 103,  77],
                         [24,  35,  55,  64,  81, 104, 113,  92],
                         [49,  64,  78,  87, 103, 121, 120, 101],
                         [72,  92,  95,  98, 112, 100, 103,  99]])
    # Cb, Cr component
    else:
        return np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                         [18, 21, 26, 66, 99, 99, 99, 99],
                         [24, 26, 56, 99, 99, 99, 99, 99],
                         [47, 66, 99, 99, 99, 99, 99, 99],
                         [99, 99, 99, 99, 99, 99, 99, 99],
                         [99, 99, 99, 99, 99, 99, 99, 99],
                         [99, 99, 99, 99, 99, 99, 99, 99],
                         [99, 99, 99, 99, 99, 99, 99, 99]])
#zigzgag for 8x8 block
def zigzag(input):
    x = 0
    y = 0
    ret = []
    bac = []
    way = 0
    for _ in range(32):
        ret.append(input[x][y])
        bac.append(input[7 - x][7 - y])
        if way == 0:
            y += 1
            way = 1
        elif way == 1:
            x += 1
            y -= 1
            if y == 0:
                way = 2
        elif way == 2:
            x += 1
            way = 3
        else:
            x -= 1
            y += 1
            if x == 0:
                way = 0
    bac.reverse()
    ret += bac
    return ret


def convert(a, b, file_name):
    global Test
    global Pil_size
    # open image and Change to YCbCr
    # find row and col feature
    im = Image.open(file_name)
    im_ycbcr = im.convert('YCbCr')
    im_np = np.asarray(im_ycbcr).astype(int)
    row, col = im_np.shape[0], im_np.shape[1]


    #find size of block
    #make array for dc, ac component
    block_size = row // 8 * col // 8
    dc = np.empty((block_size, 3), dtype=np.int32)
    ac = np.empty((block_size, 63, 3), dtype=np.int32)
    idx = 0
    out = BitStream()
    output = np.empty((3,4,4,64),dtype=np.int)
    # for 8 x 8 blocks
    for i in range(0, row, 8):
        for j in range(0, col, 8):
            # for Y, Cb, Cr
            for k in range(3):
                # divide Image to 8x8 blocks
                block = im_np[i : i + 8, j : j + 8, k] - 128


                # Test Area for uniform bpp
                block = ((np.random.beta(a, b, size = (8, 8)) * 255) - 128).astype(int)


                # Dct
                dct_block = fftpack.dct(fftpack.dct(block, norm='ortho').T, norm='ortho').T
                # quantization
                q_block = np.rint((dct_block / find_quan(k).reshape([8,8])))
                output[k][i // 8][j //  8] = q_block.flatten()
                #zigzag
                flat_block = np.array(zigzag(q_block))
                # match to dc and ac
                dc[idx, k] = flat_block[0]
                ac[idx, :, k] = flat_block[1:]
            idx += 1

    #DPCM for DC
    #Do huffman encode
    DPCM = np.empty((block_size, 3), dtype=int)
    for k in range(3):
        DPCM[0, k] = dc[0, k]
        for i in range(1, block_size):
            DPCM[i, k] = dc[i, k] - dc[i - 1, k]

    # Huffman Encoding for each MCU
    for i in range(block_size):
        for k in range(3):
            #DC
            out.write(huffmanEncode.encodeDCToBoolList(DPCM[i, k], k), bool)
            #AC wih RLE
            huffmanEncode.encodeACBlock(out, ac[i,:,k], k)
    return output, len(out) /(row * col)
