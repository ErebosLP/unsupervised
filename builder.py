import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import ipdb

from einops import rearrange


class DetCo(nn.Module):

    def __init__(self,encoder, channels =16):
        super().__init__()

        self.m = 0.999
      
        self.encoder_q =  smp.Unet(encoder_name=encoder, encoder_weights=None, classes=channels, activation='sigmoid') 
        self.encoder_k =  smp.Unet(encoder_name=encoder, encoder_weights=None, classes=channels, activation='sigmoid') 

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        batch_size = im_q.size(0)
        
        # compute query features
        
        q = self.encoder_q(im_q.type(torch.float))  # queries: NxC
        
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(im_k.type(torch.float))  # keys: NxC
            k = nn.functional.normalize(k, dim=1)
        return q,k
