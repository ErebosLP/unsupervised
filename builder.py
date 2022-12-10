import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import ipdb

from einops import rearrange


class DetCo(nn.Module):

    def __init__(self,encoder, channels =16):
        super().__init__()

        self.encoder =  smp.Unet(encoder_name=encoder, encoder_weights=None, classes=channels, activation='sigmoid') 
        
    def forward(self, im_q, im_k):

        batch_size = im_q.size(0)
        
        q = self.encoder(im_q.type(torch.float)) 
        q = nn.functional.normalize(q, dim=1)
        
        k = self.encoder(im_k.type(torch.float)) 
        k = nn.functional.normalize(k, dim=1)

        return q,k
