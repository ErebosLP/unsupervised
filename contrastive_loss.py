import torch
import numpy as np
import ipdb
class ContrastiveLoss(torch.nn.Module):
    # from paper the optimal temperature is 0.5
    def __init__(self, num_regions = 4, temperature = 0.5):
        torch.nn.Module.__init__(self)
        self.temperature = temperature
        self.loss  = torch.nn.CrossEntropyLoss()
        self.num_regions = num_regions

   
    def forward(self, views_1, views_2):#shape [1,16,256,256] -> 16,65536
        loss = 0
        for i in range(views_1.shape[0]):
            z_view1 = views_1[i].unsqueeze(0)
            z_view2 = views_2[i].unsqueeze(0)
            _, _, h, w = z_view1.size()
            h,w =32,32########################################################################################################################################################################
            neg_examples = int(h*w/self.num_regions)
            
            # Compute the Region of Intrest
            z_list_1 = []
            z_list_2 = []
            for i in range(int(np.sqrt(self.num_regions))):
                for j in range(int(np.sqrt(self.num_regions))):
                    h0,h1,w0,w1 = i/2*h, (i+1)/2*h, j/2*w, (j+1)/2*w

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
                

                patch = torch.reshape(patch,(neg_examples,16)).unsqueeze(2)
                patch_pos = torch.reshape(z_list_2[idx],(neg_examples,16)).unsqueeze(2)
                patch_neg = torch.swapaxes(torch.reshape(z_list_2[patch_neg_idx],(neg_examples,16)).unsqueeze(2),0,2)
                
                # neg_idx = np.random.choice(sample_size**2,(sample_size**2,neg_examples))
                
                patch_stack = patch.repeat(1,1,neg_examples+1)
                neg_patch_stack = torch.cat((patch_pos,patch_neg.repeat(neg_examples,1,1)),dim=2)
                
                sim = torch.nn.CosineSimilarity(dim=1, eps=1e-08)(patch_stack,neg_patch_stack)
                sim = (sim + 1)/2 # normalized to [0,1]
                sim = torch.sum(sim,dim = 0)/neg_examples
                #sim = sim / self.temperature
                
                target = torch.zeros_like(sim).cuda()
                target[0] = 1
                loss += self.loss(sim,target)
        return loss/(i+1), sim[0].detach().cpu().numpy(), sim[1:].detach().cpu().numpy().sum()/neg_examples
