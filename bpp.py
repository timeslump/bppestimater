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


def convert(file_name, writeValid):
    global Test
    global Pil_size
    
    #Test for jpeg of PIL package
    '''
    testfile = Image.open(file_name).convert('RGB')
    testfile.save('1.jpeg', 'jpeg', quality = 50, subsampling = 0)
    Pil_size.append(os.path.getsize('1.jpeg'))
    '''
    
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
    # for 8 x 8 blocks
    for i in range(0, row, 8):
        for j in range(0, col, 8):
            # for Y, Cb, Cr
            for k in range(3):
                # divide Image to 8x8 blocks
                block = im_np[i : i + 8, j : j + 8, k] - 128
                
                
                # Test Area for uniform bpp 
                '''
                # just random range
                for i in range(8):
                    for j in range(8):
                        block[i][j] = random.randrange(0, 256)
                #beta distribution
                block = ((np.random.beta(1, 5, size = (8, 8)) * 255) - 128).astype(int)
                '''
                
                # Dct
                dct_block = fftpack.dct(fftpack.dct(block, norm='ortho').T, norm='ortho').T
                # quantization
                q_block = np.rint((dct_block / find_quan(k).reshape([8,8])))
                #zigzag
                flat_block = np.array(zigzag(q_block))
                
                # match to dc and ac
                dc[idx, k] = flat_block[0]
                ac[idx, :, k] = flat_block[1:]
                '''
                # for test area
                test_ac = np.array([abs( (i % 3) - 3) for i in range(64)])
                test_dc = random.randrange(1, 5)
                dc[idx, k] = test_dc
                test_ac = np.array([abs( (i % 3) - 3) for i in range(64)])
                ac[idx, :, k] = test_ac[1:]
                '''
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
            
    output[round(len(out) * 10 /(row * col))] += 1
    
    # show bpp by line
    #print(len(out) /(row * col))
    
    # if want to Make output file
    if writeValid:
        writefile(out, file_name, row, col)
        
#write file with header 
def writefile(out, file_name, row, col):
    global Make_size
    jpegFile = open(file_name + '2jpeg.jpeg', 'wb+')
    # write jpeg header
    jpegFile.write(huffmanEncode.hexToBytes('FFD8FFE000104A46494600010100000100010000'))
    # write y Quantization Table
    jpegFile.write(huffmanEncode.hexToBytes('FFDB004300'))
    luminanceQuantTbl = find_quan(0).reshape([64])
    jpegFile.write(bytes(luminanceQuantTbl.tolist()))
    # write u/v Quantization Table
    jpegFile.write(huffmanEncode.hexToBytes('FFDB004301'))
    chrominanceQuantTbl = find_quan(1).reshape([64])
    jpegFile.write(bytes(chrominanceQuantTbl.tolist()))
    # write height and width
    jpegFile.write(huffmanEncode.hexToBytes('FFC0001108'))
    hHex = hex(col)[2:]
    while len(hHex) != 4:
        hHex = '0' + hHex

    jpegFile.write(huffmanEncode.hexToBytes(hHex))

    wHex = hex(row)[2:]
    while len(wHex) != 4:
        wHex = '0' + wHex

    jpegFile.write(huffmanEncode.hexToBytes(wHex))

    # 03    01 11 00    02 11 01    03 11 01
    # 1：1	01 11 00	02 11 01	03 11 01
    # 1：2	01 21 00	02 11 01	03 11 01
    # 1：4	01 22 00	02 11 01	03 11 01
    # write Subsamp
    jpegFile.write(huffmanEncode.hexToBytes('03011100021101031101'))

    #write huffman table
    jpegFile.write(huffmanEncode.hexToBytes('FFC401A20000000701010101010000000000000000040503020601000708090A0B0100020203010101010100000000000000010002030405060708090A0B1000020103030204020607030402060273010203110400052112314151061361227181143291A10715B14223C152D1E1331662F0247282F12543345392A2B26373C235442793A3B33617546474C3D2E2082683090A181984944546A4B456D355281AF2E3F3C4D4E4F465758595A5B5C5D5E5F566768696A6B6C6D6E6F637475767778797A7B7C7D7E7F738485868788898A8B8C8D8E8F82939495969798999A9B9C9D9E9F92A3A4A5A6A7A8A9AAABACADAEAFA110002020102030505040506040803036D0100021103042112314105511361220671819132A1B1F014C1D1E1234215526272F1332434438216925325A263B2C20773D235E2448317549308090A18192636451A2764745537F2A3B3C32829D3E3F38494A4B4C4D4E4F465758595A5B5C5D5E5F5465666768696A6B6C6D6E6F6475767778797A7B7C7D7E7F738485868788898A8B8C8D8E8F839495969798999A9B9C9D9E9F92A3A4A5A6A7A8A9AAABACADAEAFA'))
    # SOS Start of Scan
    # yDC yAC uDC uAC vDC vAC
    sosLength = out.__len__()
    filledNum = 8 - sosLength % 8
    if(filledNum!=0):
        out.write(np.ones([filledNum]).tolist(),bool)

    jpegFile.write(bytes([255, 218, 0, 12, 3, 1, 0, 2, 17, 3, 17, 0, 63, 0])) # FF DA 00 0C 03 01 00 02 11 03 11 00 3F 00

    # write encoded data
    sosBytes = out.read(bytes)
    for i in range(len(sosBytes)):
        jpegFile.write(bytes([sosBytes[i]]))
        if(sosBytes[i]==255):
            jpegFile.write(bytes([0])) # FF to FF 00

    # write end symbol
    jpegFile.write(bytes([255,217])) # FF D9
    jpegFile.close()
    Make_size.append(os.path.getsize(file_name + '2jpeg.jpeg'))
    
 
output = [0] * 250
Pil_size = []
Make_size =[]

if __name__ == '__main__':
    writeValid = False
    
    # folder with contain images
    files = os.listdir(r'C:\Users\Jin\untitled\testSet')
    # for each image in folder
    for i, file in enumerate(files):
        convert(r'C:\Users\Jin\untitled\testSet\\' + file, writeValid)

    #plot bpp by range
    x = np.arange(250)
    x_bar = [i*0.1 if i % 50 == 0 else '' for i in range(0, 250)]
    plt.bar(x, output)
    plt.xticks(x, x_bar)
    #plt.scatter(Pil_size, Make_size)
    plt.show()

