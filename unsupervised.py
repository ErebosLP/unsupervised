import os
import torch
import ipdb
import City_imageloader
import builder
import torchvision.transforms as transforms
import random
import time
import numpy as np
import contrastive_loss
from torch.utils.tensorboard import SummaryWriter
import utils
import augmentation as aug

def main():
    #Hyperparameter    
    numEpochs = 100
    learningRate = 0.0001

    numImgs = 3000
    batchsize = 50
    numClasses = 16
    print_freq = int(50)
    encoder = 'resnet50'
    
    model_name = 'model_DetCo_' + encoder + '_numEpochs_' + str(numEpochs)+ '_lr_0_' + str(learningRate)[-3:] + '_batch_' + str(batchsize)
    img_path = '/media/jean-marie/WD_BLACK/Datasets/'
    out_dir = './results/' + model_name
    
    start_saving =  numEpochs/2 #when to start saving the max_valid_model

    

    if not os.path.exists(os.path.join(out_dir,'model/checkpoint')):
        os.makedirs(os.path.join(out_dir,'model/checkpoint'))
        
    if not os.path.exists(os.path.join(out_dir,'runs')):
        os.makedirs(os.path.join(out_dir,'runs'))

    writer = SummaryWriter(os.path.join(out_dir,'runs'))

    min_trainLoss = np.inf

    model = builder.DetCo(encoder,numClasses)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    augaugmentation = [
            #transforms.RandomResizedCrop(256, scale=(0.6, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            #transforms.RandomHorizontalFlip(),
            aug.GaussianBlur(1,np.random.uniform(0.1,2)),
            transforms.RandomApply([aug.Sobel()],p=1),
            transforms.ToTensor(),
            #normalize
        ]
    #checkpoint = torch.load('./checkpoint_0150.pth.tar')
    Citydataset = City_imageloader.CityscapeDataset(img_path, 'val',City_imageloader.TwoCropsTransform(transforms.Compose(augmentation)),num_imgs=numImgs)
    numImgs = Citydataset.__len__()
    #model.load_state_dict(checkpoint['state_dict'])

    # #########################################
    # import matplotlib.pyplot as plt
    # img_id =29
    # image = Citydataset[img_id][0]
    # test1,perm = _jigsaw(image.unsqueeze(0),4,4)
    # test2 = _jigsaw_backwards(test1,perm)
    
    
    # test1 = (test1.squeeze(0).numpy().transpose(1, 2, 0)*255).astype(np.uint8)
    # test2 = (test2.squeeze(0).numpy().transpose(1, 2, 0)*255).astype(np.uint8)
    # plt.imsave('./input.jpg',(image.numpy().transpose(1, 2, 0)*255).astype('uint8'))
    # plt.imsave('./output.jpg',test1.astype('uint8'))
    # plt.imsave('./output1.jpg',test2.astype('uint8'))
    # ipdb.set_trace()
    ##########################################

    model.to('cuda')
    model.eval()
    train_loader = torch.utils.data.DataLoader(Citydataset, batch_size=batchsize, shuffle=True, num_workers=1, pin_memory=True, drop_last=False)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=learningRate, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, numEpochs)

    loss = contrastive_loss.ContrastiveLoss()
    t1 = time.time()
    for epoch in range(numEpochs):
        torch.cuda.empty_cache()
        model.train()
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Epoch: [{}]'.format(epoch)
        losses = np.array([0.0,0.0,0.0,0.0])
        print('start train epoch: ' + str(epoch))
        for idx, (view_1,view_2) in enumerate(metric_logger.log_every(train_loader,print_freq,header)):
            
            view_1 =view_1.cuda()
            view_2 =view_2.cuda()
            
            q,k = model(im_q=view_1, im_k=view_2)
            
            view_1_jig ,view_1_perm = aug._jigsaw(view_1)
            view_2_jig ,view_2_perm = aug._jigsaw(view_2)
            q_jig, k_jig = model(im_q=view_1_jig, im_k=view_2_jig)
            q_jig = aug._jigsaw_backwards(q_jig,view_1_perm)
            k_jig = aug._jigsaw_backwards(k_jig,view_2_perm)

            batch_loss_g2g, pos_g2g, neg_g2g = loss(q,k)
            batch_loss_l2l, pos_l2l, neg_l2l = loss(q_jig,k_jig)
            batch_loss_g2l, pos_g2l, neg_g2l = loss(q_jig,k)
            batch_loss = batch_loss_g2g +batch_loss_l2l +  batch_loss_g2l

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()     

            metric_logger.update(loss=batch_loss)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

            losses += [batch_loss.detach().cpu().numpy() ,batch_loss_g2g.detach().cpu().numpy(),batch_loss_l2l.detach().cpu().numpy(),batch_loss_g2l.detach().cpu().numpy()]
        scheduler.step()  
        losses /= idx+1
        writer.add_scalars('Loss',  {'batch loss':losses[0],'global loss':losses[1],'local loss':losses[2] ,'global2local loss':losses[3]}, epoch)
        writer.add_scalars('similarity',  {'pos_g2g':pos_g2g,'pos_l2l':pos_l2l,'pos_g2l':pos_g2l ,'neg_g2g':neg_g2g,'neg_l2l':neg_l2l,'neg_g2l':neg_g2l }, epoch)

                # update the learning rate
        if epoch % 15 == 0:
            torch.save(model.state_dict(), out_dir + '/model/checkpoint/%08d_model.pth' % (epoch))
            torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                # 'loss': loss
                }, out_dir + '/model/checkpoint/%08d_model.pth' % (epoch))

        print('losses & min_trainLoss', losses[0] ,'/', min_trainLoss)
        if epoch > start_saving:
            if min_trainLoss > losses[0]:
                min_trainLoss = losses[0]
                # torch.save(model.state_dict(), out_dir + '/checkpoint/max_valid_model.pth')
                torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    # 'loss': loss
                    }, os.path.join(out_dir,'model/'+ 'max_valid_model.pth'))
        time_estimate = (time.time() - t1) / (epoch + 1) * (numEpochs - (epoch + 1))
        
        print(f'Estimated time until finished: Days: {int(time_estimate/(24*3600)) }, Hours: {int(time_estimate/3600) % 24}, Minutes: {int(time_estimate/60) % 60}, Seconds: {int(time_estimate % 60)}')
    torch.save(model.state_dict(), os.path.join(out_dir,'model/'+ model_name + '.pth'))



if __name__=='__main__':
    main()