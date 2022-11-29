import torch
import random
import numpy as np
from PIL import ImageFilter
from scipy import ndimage


def _random_crop(x, ch=64, cw=64):
    #import random
    _, _, h, w = x.size()
    assert h >= ch and w >= cw, f'crop error: {h} or {w} is not larger than {ch} or {cw}'

    i = random.randint(0, h - ch)
    j = random.randint(0, w - cw)

    return x[:, :, i:i+ch, j:j+cw]

class GaussianBlur(object):
    def __init__(self, p,alpha):
        self.p = p
        self.alpha = alpha
        
    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * self.alpha + 0.1
            
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img

class Sobel:
    def __init__(self):
        pass
        
    def __call__(self,img):
        # Get x-gradient in "sx"

        sx = ndimage.sobel(img,axis=0,mode='constant')
        # Get y-gradient in "sy"
        sy = ndimage.sobel(img,axis=1,mode='constant')
        # Get square root of sum of squares
        sobel_out=np.hypot(sx,sy)
            
        return sobel_out


def _jigsaw( x, nh=4, nw=4): #Use 2^x as #patches per row or implement for decimal numbers of pixels
    #import random
    _, _, h, w = x.size()
    assert h % nh == 0 and w % nw == 0, f'jigsaw error: {h} or {w} is not divisible by {nh} or {nw}'

    x_list = []
    for i in range(nh):
        for j in range(nw):
            h0,h1,w0,w1 = i/nh*h, (i+1)/nh*h, j/nw*w, (j+1)/nw*w
            patch = x[:, :, int(h0):int(h1), int(w0):int(w1)]
            x_list += [patch]
    permutation = torch.randperm(nh*nw)
    permutation = torch.reshape(permutation,(nh,nw))
    new_img = torch.zeros_like(x)
    
    for i in range(nh):
        for j in range(nw):
            h0,h1,w0,w1 = i/nh*h, (i+1)/nh*h, j/nw*w, (j+1)/nw*w
            new_img[:, :, int(h0):int(h1), int(w0):int(w1)] = x_list[permutation[i,j]]

    return new_img, permutation

def _jigsaw_backwards(x, permutation):
    import ipdb
    
    nh,nw = permutation.size()
    _, _, h, w = x.size()
    assert h % nh == 0 and w % nw == 0, f'jigsaw error: {h} or {w} is not divisible by {nh} or {nw}'

    x_list = []
    for i in range(nh):
        for j in range(nw):
            h0,h1,w0,w1 = i/nh*h, (i+1)/nh*h, j/nw*w, (j+1)/nw*w
            patch = x[:, :, int(h0):int(h1), int(w0):int(w1)]
            x_list += [patch]
    new_img = torch.zeros_like(x)
    permutation = torch.reshape(permutation,(nh*nw,))
    for i in range(nh):
        for j in range(nw):
            index = i*nh+j
            patch_idx = permutation[index]
            #j = patch_idx % nh
            #i = int(patch_idx/nw)

            h0,h1,w0,w1 = int(patch_idx/nw)/nh*h, (int(patch_idx/nw)+1)/nh*h, (patch_idx % nh)/nw*w, ((patch_idx % nh)+1)/nw*w
            
            new_img[:, :, int(h0):int(h1), int(w0):int(w1)] = x_list[index] 
    return new_img