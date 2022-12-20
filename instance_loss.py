import torch
import numpy as np
import ipdb
class InstanceLoss(torch.nn.Module):
    # from paper the optimal temperature is 0.5
    def __init__(self, ):
        torch.nn.Module.__init__(self)
        self.loss  = torch.nn.BCELoss()
        

   
    def forward(self, views_1, views_2, target):#shape [1,16,256,256] -> 16,65536
        instance_sim = torch.zeros(11).cuda()
        class_sim = torch.zeros(11).cuda()
        class_std = torch.zeros((11)).cuda()
        neg_sim = torch.zeros(1).cuda()

        instances_view1 = []
        instances_view2 = []
        classes = []

        for k in range(target['labels'].shape[1]):
            mask = target['masks'][0][k]
            classes += [target['labels'][0][k]]
            count_neg_instances = torch.zeros(1).cuda()
            instance_pixels = views_1[0,:,mask==1]
            instance_pixels = torch.permute(instance_pixels,(1,0))
            instances_view1 += [instance_pixels]
            instance_pixels = views_2[0,:,mask==1]
            instance_pixels = torch.permute(instance_pixels,(1,0))
            instances_view2 += [instance_pixels]

        for instance in range(len(instances_view1)):
            mean = torch.mean(instances_view1[instance],dim=0)
            class_std[classes[instance]] += torch.std(instances_view1[instance])
            for compare in range(len(instances_view1)):
                if (instance == compare):
                    mean_vec = torch.ones(instances_view2[compare].shape).cuda() * mean
                    sim = torch.nn.CosineSimilarity(dim=1,eps=1e-8)(mean_vec,instances_view2[compare])
                    instance_sim[classes[instance]] += sim.mean()
                elif(classes[instance] == classes[compare]):
                    mean_vec = torch.ones(instances_view2[compare].shape).cuda() * mean
                    sim = torch.nn.CosineSimilarity(dim=1,eps=1e-8)(mean_vec,instances_view2[compare])
                    class_sim[classes[instance]] += sim.mean()
                else:
                    mean_vec = torch.ones(instances_view2[compare].shape).cuda() * mean
                    sim = torch.nn.CosineSimilarity(dim=1,eps=1e-8)(mean_vec,instances_view2[compare])
                    neg_sim += sim.mean()
                    count_neg_instances += 1
        
        neg_sim /= count_neg_instances
        for class_ in np.unique(classes):
            class_sum = torch.tensor(np.sum(classes == class_)).cuda()
            if class_sum > 1:
                instance_sim[class_] /= class_sum
                class_sim[class_] /= class_sum * (class_sum - 1)
                class_std[class_] /= class_sum
        return instance_sim, class_sim, neg_sim,class_std
