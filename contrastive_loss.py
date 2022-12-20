import torch
import numpy as np
import ipdb
class ContrastiveLoss(torch.nn.Module):
    # from paper the optimal temperature is 0.5
    def __init__(self, num_regions = 4, temperature = 2):
        torch.nn.Module.__init__(self)
        self.temperature = temperature
        self.loss  = torch.nn.BCELoss()
        self.num_regions = num_regions

   
    def forward(self, views_1, views_2):#shape [1,16,256,256] -> 16,65536
        loss = 0
        sim_all = torch.zeros((int((views_1.size()[2] * views_1.size()[3])/self.num_regions) + 1)).cuda()
        for i in range(views_1.shape[0]):
            z_view1 = views_1[i].unsqueeze(0)
            z_view2 = views_2[i].unsqueeze(0)
            _, _, h, w = z_view1.size()
            #h,w =32,32########################################################################################################################################################################
            neg_examples = int(h*w/self.num_regions)
            
            # Compute the Region of Intrest
            z_list_1 = []
            z_list_2 = []
            patches_per_row = int(np.sqrt(self.num_regions))
            for l in range(int(np.sqrt(self.num_regions))):
                for j in range(int(np.sqrt(self.num_regions))): 
                    h0,h1,w0,w1 = l/patches_per_row*h, (l+1)/patches_per_row*h, j/patches_per_row*w, (j+1)/patches_per_row*w

                    # Patch from view 1
                    patch = z_view1[:, :, int(h0):int(h1), int(w0):int(w1)]
                    patch = (patch - patch.mean()) / patch.std()
                    z_list_1 += [patch]

                    # Patch from view 2
                    patch = z_view2[:, :, int(h0):int(h1), int(w0):int(w1)]
                    patch = (patch - patch.mean()) / patch.std()
                    z_list_2 += [patch]
            
            for idx , patch in enumerate(z_list_1):
                patch_neg_idx = self.num_regions - 1 - idx
            
                patch = torch.reshape(patch,(neg_examples,views_1.shape[1])).unsqueeze(2)
                patch_pos = torch.reshape(z_list_2[idx],(neg_examples,views_1.shape[1])).unsqueeze(2)
                patch_neg = torch.swapaxes(torch.reshape(z_list_2[patch_neg_idx],(neg_examples,views_1.shape[1])).unsqueeze(2),0,2)
                
                # neg_idx = np.random.choice(sample_size**2,(sample_size**2,neg_examples))
                
                patch_stack = patch.repeat(1,1,neg_examples+1)
                neg_patch_stack = torch.cat((patch_pos,patch_neg.repeat(neg_examples,1,1)),dim=2)
                
                sim = torch.nn.CosineSimilarity(dim=1, eps=1e-08)(patch_stack,neg_patch_stack)
                #sim = (sim + 1)/2 # normalized to [0,1]
                sim = torch.nn.ReLU(sim)
                sim = torch.sum(sim,dim = 0)/neg_examples
                sim[1:] = sim[1:] / self.temperature
                sim_all += sim
                target = torch.zeros_like(sim).cuda()
                target[0] = 1
                loss += self.loss(sim,target)
                sim_all.detach().cpu().numpy()
        return loss / (i+1) / (idx+1), sim_all[0] / (i+1) / (idx+1), sim_all[1:].sum() / neg_examples * self.temperature / (i+1) / (idx+1)
