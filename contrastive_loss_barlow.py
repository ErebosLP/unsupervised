import torch
import numpy as np
import ipdb
class ContrastiveLoss(torch.nn.Module):
    # from paper the optimal temperature is 0.5
    def __init__(self, temperature = 2, factor = 0.8,examples = 200*200):
        torch.nn.Module.__init__(self)
        self.temperature = temperature
        self.loss  = torch.nn.BCELoss()
        self.factor = factor
        self.examples = examples
   
    def forward(self, views_1, views_2,img):
        torch.cuda.empty_cache()
        img = img.to('cuda')
        loss = 0
        pos_corr = 0
        neg_corr = 0
        #sim_all = torch.zeros((self.neg_examples + 1)).cuda()
        batch, c, h, w = views_1[0].unsqueeze(0).size()
        height  = np.floor(np.arange(h*w)/w).astype(int)
        width = np.floor(np.arange(h*w)%w).astype(int)
        max_euc_dist = torch.norm(torch.tensor([256,256],dtype=float).to('cuda'))
        max_rgb_dist = torch.sqrt(torch.tensor([3.]).to('cuda'))

        view_1_norm = torch.nn.functional.normalize(views_1,p=2,dim=1)
        view_2_norm = torch.nn.functional.normalize(views_2,p=2,dim=1)
        for i in range(views_1.shape[0]):
            
            z_view1 = view_1_norm[i].unsqueeze(0)
            z_view2 = view_2_norm[i].unsqueeze(0)
            ########################################################################################################################################################################
            ########################################################################################################################################################################
            z_view1_vec = torch.reshape(z_view1,[batch,c,h*w]).squeeze(0).unsqueeze(2)
            z_view2_vec = torch.reshape(z_view2,[batch,c,h*w]).squeeze(0).unsqueeze(2)

            idx = np.zeros(z_view1_vec.shape[1], bool)
            idx[:self.examples] = 1
            idx =  np.random.permutation(idx)
            
                
            z_view1_vec = z_view1_vec[:,idx,:]
            z_view2_vec = z_view2_vec[:,idx,:]

            mat =  z_view1_vec.squeeze(2).T @ z_view2_vec.squeeze(2)
            
            identity_stacked_matrix = torch.eye(self.examples).cuda()

            loss += (mat -identity_stacked_matrix).pow(2).sum()/(self.examples**2)
            
            pos_corr +=  torch.diagonal(mat).sum()/self.examples
            neg_corr += (mat.sum() - (pos_corr*self.examples))/(self.examples*(self.examples-1))

        return loss / (i+1), pos_corr / (i+1), neg_corr / (i+1)
