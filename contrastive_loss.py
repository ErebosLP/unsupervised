import torch
import numpy as np
import ipdb
class ContrastiveLoss(torch.nn.Module):
    # from paper the optimal temperature is 0.5
    def __init__(self, temperature = 2, factor = 0.8,neg_examples = 256):
        torch.nn.Module.__init__(self)
        self.temperature = temperature
        self.loss  = torch.nn.BCELoss()
        self.factor = factor
        self.neg_examples = neg_examples
   
    def forward(self, views_1, views_2,img):
        loss = 0
        sim_all = torch.zeros((self.neg_examples + 1)).cuda()
        for i in range(views_1.shape[0]):
            z_view1 = views_1[i].unsqueeze(0)
            z_view2 = views_2[i].unsqueeze(0)
            ########################################################################################################################################################################
            
            ########################################################################################################################################################################
            
            batch, c, h, w = z_view1.size()
            
            
            z_view1_vec = torch.reshape(z_view1,[batch,c,h*w]).squeeze(0).unsqueeze(2)
            
            neg_idx = np.array([np.random.choice(h,(h*w,self.neg_examples)),np.random.choice(w,(h*w,self.neg_examples))])
            patch_neg = z_view2[:,:,neg_idx[0],neg_idx[1]].squeeze(0)   

            euc_dist = torch.zeros([h*w,self.neg_examples]).to('cuda')
            rgb_dist = torch.zeros([h*w,self.neg_examples]).to('cuda')
            for idx in range(z_view1_vec.shape[1]):
                height = int(idx/w)
                width = idx%w
                for j in range(self.neg_examples):
                    euc_dist[idx,j] = torch.norm(torch.tensor([height,width],dtype=float).to('cuda')-torch.tensor([neg_idx[0,idx,j],neg_idx[1,idx,j]],dtype=float).to('cuda'))
                    rgb_dist[idx,j] = torch.norm(img[0,:,height,width] - img[0,:,neg_idx[0,idx,j],neg_idx[1,idx,j]])
            euc_dist /= torch.norm(torch.tensor([h-1,w-1],dtype=float).to('cuda'))
            rgb_dist /= torch.sqrt(torch.tensor([3.]).to('cuda'))
            weight = euc_dist * self.factor + rgb_dist * (1-self.factor)
            patch_stack = z_view1_vec.repeat(1,1,self.neg_examples+1)

            neg_patch_stack = torch.cat((z_view1_vec,patch_neg),dim=2)
            sim = torch.nn.CosineSimilarity(dim=0, eps=1e-08)(patch_stack,neg_patch_stack)
            sim[:,1:] *= weight
            #sim = (sim + 1)/2 # normalized to [0,1]
            #sim = torch.nn.ReLU()(sim)
            sim = torch.abs(sim)
            sim[sim>=1] = 1
            sim = torch.sum(sim,dim = 0)/(h*w)
            sim[1:] = sim[1:] / self.temperature
            sim_all += sim
            target = torch.zeros_like(sim).cuda()
            target[0] = 1
            loss += self.loss(sim,target)
            sim_all.detach().cpu().numpy()
        return loss / (i+1), sim_all[0] / (i+1) , sim_all[1:].sum() / self.neg_examples * self.temperature / (i+1) 
