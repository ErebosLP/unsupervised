import torch
import numpy as np
import ipdb
class InstanceLoss(torch.nn.Module):
    # from paper the optimal temperature is 0.5
    def __init__(self, ):
        torch.nn.Module.__init__(self)
        self.loss  = torch.nn.BCELoss()
        

   
    def forward(self, views_1, views_2, target):#shape [1,16,256,256] -> 16,65536
        instance_sim = torch.zeros(11)
        class_sim = torch.zeros(11)
        class_std = torch.zeros(11)
        neg_sim = 0
        for i in range(views_1.shape[0]):

            instances_view1 = []
            instances_view2 = []
            classes = []

            for k in range(target[i]['labels']):
                mask = target[i]['masks'][k]
                classes += [target[i]['labels'][k]]
                count_neg_instances = 0
                instance_pixels = views_1[:,mask==1]
                instance_pixels = torch.permute(instance_pixels,(1,0)).unsqueeze(1)
                instances_view1 += [instance_pixels]
                instance_pixels = views_2[:,mask==1]
                instance_pixels = torch.permute(instance_pixels,(1,0)).unsqueeze(1)
                instances_view2 += [instance_pixels]

            for instance in range(len(instances_view1)):
                mean = torch.mean(instances_view1[instance],dim=0)
                class_std[classes[instance]] += torch.std(instances_view1[instance],dim=0)
                for compare in range(len(instances_view1)):
                    if (instance == compare):
                        mean_vec = torch.ones([instances_view2[compare].shape]) * mean
                        sim = torch.nn.CosineSimilarity(dim=1,eps=1e-8)(mean_vec,instances_view2[compare])
                        instance_sim[classes[instance]] += sim.mean()
                    elif(classes[instance] == classes[compare]):
                        mean_vec = torch.ones([instances_view2[compare].shape]) * mean
                        sim = torch.nn.CosineSimilarity(dim=1,eps=1e-8)(mean_vec,instances_view2[compare])
                        class_sim[classes[instance]] += sim.mean()
                    else:
                        mean_vec = torch.ones([instances_view2[compare].shape]) * mean
                        sim = torch.nn.CosineSimilarity(dim=1,eps=1e-8)(mean_vec,instances_view2[compare])
                        neg_sim += sim.mean()
                        count_neg_instances += 1

            neg_sim /= count_neg_instances
            for class_ in range(len(classes)):
                instance_sim[class_] /= torch.sum(classes == class_)
                class_sim[class_] /= torch.sum(classes == class_) * (torch.sum(classes == class_) - 1)
                class_std[class_] /= torch.sum(classes == class_)

            
        return instance_sim, class_sim, neg_sim