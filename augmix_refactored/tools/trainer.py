import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from tqdm.autonotebook import tqdm

from augmix_refactored.config import Config


def train(net, train_loader, optimizer, scheduler, config: Config, logging, epoch):
    """Train for one epoch."""
    net.train()
    loss_ema = 0.
    iter = 0
    epsilon = 1e-20
    with tqdm(train_loader, unit="batch", disable=config.disable_tqdm) as tepoch:
        for images, targets in tepoch:
        #for i, (images, targets) in enumerate(train_loader):
            optimizer.zero_grad()

            if config.no_jsd:
                images = images.cuda()
                targets = targets.cuda()
                logits = net(images)
                loss = F.cross_entropy(logits, targets, reduction=config.reduction)
                
                if config.sigmoid:
                    sig_clean = F.sigmoid(logits)
                elif config.softmax:
                    sig_clean = F.softmax(logits, dim=1)
                if config.cossim:
                    one_hot_tar = F.one_hot(targets, num_classes=10 if config.dataset=='cifar10' else 100).cuda()
                    #one_hot_tar = torch.zeros(sig_clean.shape).cuda()
                    #i = 0
                    #for tar in targets:
                    #    one_hot_tar[i][tar] = 1
                    #    i +=1
                    # OPTION 1                    
                    # sim = F.cosine_similarity(sig_clean, one_hot_tar, dim=0)                          
                    # OPTION 2                    
                    sim = F.cosine_similarity(sig_clean, one_hot_tar, dim=1)   
                    loss = ((1-sim) * loss).mean()
                if config.l2:
                    one_hot_tar = F.one_hot(targets, num_classes=10 if config.dataset=='cifar10' else 100).cuda().to(dtype=sig_clean.dtype)
                    #import ipdb;ipdb.set_trace()
                    #one_hot_tar = torch.zeros(sig_clean.shape).cuda()
                    #i = 0
                    #for tar in targets:
                    #    one_hot_tar[i][tar] = 1
                    #    i +=1
                    # OPTION 1                    
                    # sim = torch.cdist(sig_clean, one_hot_tar)                          
                    # OPTION 2                    
                    sim = torch.cdist(sig_clean, one_hot_tar)  
                    
                    loss = (sim * loss).mean() 
                if config.mse:
                    one_hot_tar = torch.zeros(sig_clean.shape).cuda()
                    i = 0
                    for tar in targets:
                        one_hot_tar[i][tar] = 1
                        i +=1                 
                    sim = F.mse_loss(sig_clean, one_hot_tar)  
                    loss = loss + sim         
                if config.jsd_scale: 
                    #one_hot_tar = torch.zeros(sig_clean.shape).cuda()
                    #i = 0
                    #for tar in targets:
                    #    one_hot_tar[i][tar] = 1
                    #    i +=1                 
                    one_hot_tar = F.one_hot(targets, num_classes=10 if config.dataset=='cifar10' else 100).cuda().to(dtype=sig_clean.dtype)
                    m_sig_clean = 0.5*(sig_clean + one_hot_tar)
                    #import ipdb;ipdb.set_trace()
                    sim = 0.5*(F.kl_div(m_sig_clean, sig_clean, reduction="batchmean")) + 0.5*(F.kl_div(m_sig_clean, one_hot_tar, reduction="batchmean"))
                    #sim = 0.5*(F.kl_div(sig_clean, m_sig_clean, reduction="batchmean")) + 0.5*(F.kl_div(one_hot_tar, m_sig_clean, reduction="batchmean"))
                    if config.only_jsd_scale:
                        loss = 12*(sim**2)
                    else:
                        loss = loss + 12*(sim**2)                   
            else:
                images_all = torch.cat(images, 0).cuda()
                targets = targets.cuda()
                logits_all = net(images_all)
                logits_clean, logits_aug1, logits_aug2 = torch.split(
                    logits_all, images[0].size(0))

                # Cross-entropy is only computed on clean images
                mean_loss = F.cross_entropy(logits_clean, targets, reduction='mean')
                loss = F.cross_entropy(logits_clean, targets, reduction=config.reduction)

                p_clean, p_aug1, p_aug2 = F.softmax(
                    logits_clean, dim=1), F.softmax(
                        logits_aug1, dim=1), F.softmax(
                            logits_aug2, dim=1)

                # Clamp mixture distribution to avoid exploding KL divergence
                p_mixture = torch.clamp(
                    (p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
                mean_loss += 12 * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                            F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                            F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.
                
                if config.sigmoid:
                    sig_clean, sig_aug1, sig_aug2 = F.sigmoid(
                            logits_clean), F.sigmoid(
                                logits_aug1), F.sigmoid(
                                    logits_aug2)                
                elif config.softmax:
                    sig_clean, sig_aug1, sig_aug2 = p_clean, p_aug1, p_aug2

                if config.sim:
                    sim = []
                    #import ipdb;ipdb.set_trace()
                    for tar, p_c, p_a1, p_a2 in zip(targets, sig_clean, sig_aug1, sig_aug2):                
                        p_c = p_c/p_c[tar]
                        p_a1 = p_a1/p_a1[tar]
                        p_a2 = p_a2/p_a2[tar]                    
                        sim.append((p_c + p_a1 + p_a2)/3)

                    #import ipdb;ipdb.set_trace()
                    mean_loss += (torch.stack(sim) * loss).mean(dim=1).mean()
                if config.cossim:
                    #one_hot_tar = torch.zeros(sig_clean.shape).cuda()
                    #i = 0
                    #for tar in targets:
                    #    one_hot_tar[i][tar] = 1
                    #    i +=1
                    one_hot_tar = F.one_hot(targets, num_classes=10 if config.dataset=='cifar10' else 100).cuda().to(dtype=sig_clean.dtype)
                    # OPTION 1
                    """
                    p_c = F.cosine_similarity(sig_clean, one_hot_tar, dim=0)
                    p_a1 = F.cosine_similarity(sig_aug1, one_hot_tar, dim=0)
                    p_a2 = F.cosine_similarity(sig_aug2, one_hot_tar, dim=0)        
                    """

                    # OPTION 2
                    #"""
                    p_c = F.cosine_similarity(sig_clean, one_hot_tar, dim=1)                    
                    p_a1 = F.cosine_similarity(sig_aug1, one_hot_tar, dim=1)
                    p_a2 = F.cosine_similarity(sig_aug2, one_hot_tar, dim=1)        
                    #"""
                    sim=((1-p_c) + (1-p_a1) + (1-p_a2))/3
                    #sim = []                
                    #import ipdb;ipdb.set_trace()
                    #for tar, p_c, p_a1, p_a2 in zip(targets, sig_clean, sig_aug1, sig_aug2):                
                    #    one_hot_tar = torch.zeros(p_c.shape).cuda()
                    #    one_hot_tar[tar] = 1
                    #    p_c = F.cosine_similarity(p_c, one_hot_tar, dim=0)
                    #    p_a1 = F.cosine_similarity(p_a1, one_hot_tar, dim=0)
                    #    p_a2 = F.cosine_similarity(p_a2, one_hot_tar, dim=0)        
                    #    sim.append((p_c + p_a1 + p_a2)/3)

                    #import ipdb;ipdb.set_trace()
                    #loss = (torch.stack(sim) * loss).mean()
                    #loss = (torch.stack(sim) * loss).sum()

                    #loss = (sim * loss).sum() # OPTION 1
                    mean_loss += (sim * loss).mean()
                if config.l2:
                    #one_hot_tar = torch.zeros(sig_clean.shape).cuda()
                    #i = 0
                    #for tar in targets:
                    #    one_hot_tar[i][tar] = 1
                    #    i +=1
                    one_hot_tar = F.one_hot(targets, num_classes=10 if config.dataset=='cifar10' else 100).cuda().to(dtype=sig_clean.dtype)

                    p_c = torch.cdist(sig_clean, one_hot_tar)
                    p_a1 = torch.cdist(sig_aug1, one_hot_tar)
                    p_a2 = torch.cdist(sig_aug2, one_hot_tar)        
                    sim=(p_c + p_a1 + p_a2)/3    
                    #import ipdb;ipdb.set_trace()                
                    mean_loss += (sim * loss).mean() # OPTION 2
                if config.mse:
                    #one_hot_tar = torch.zeros(sig_clean.shape).cuda()
                    #i = 0
                    #for tar in targets:
                    #    one_hot_tar[i][tar] = 1
                    #    i +=1
                    one_hot_tar = F.one_hot(targets, num_classes=10 if config.dataset=='cifar10' else 100).cuda().to(dtype=sig_clean.dtype)
                    
                    p_c = F.mse_loss(sig_clean, one_hot_tar)
                    p_a1 = F.mse_loss(sig_aug1, one_hot_tar)
                    p_a2 = F.mse_loss(sig_aug2, one_hot_tar)        
                    sim=(p_c + p_a1 + p_a2)/3                    

                    mean_loss += sim + loss
                if config.jsd_scale:
                    #one_hot_tar = torch.zeros(sig_clean.shape).cuda()
                    #i = 0
                    #for tar in targets:
                    #    one_hot_tar[i][tar] = 1
                    #    i +=1
                    one_hot_tar = F.one_hot(targets, num_classes=10 if config.dataset=='cifar10' else 100).cuda().to(dtype=sig_clean.dtype)
                    
                    m_clean = 0.5*(sig_clean + one_hot_tar)
                    m_aug1 = 0.5*(sig_aug1 + one_hot_tar)
                    m_aug2 = 0.5*(sig_aug2 + one_hot_tar)



                    p_c = 0.5*(F.kl_div(m_clean, sig_clean, reduction='batchmean')) + 0.5*(F.kl_div(m_clean, one_hot_tar, reduction='batchmean'))
                    p_a1 = 0.5*(F.kl_div(m_aug1, sig_aug1, reduction='batchmean')) + 0.5*(F.kl_div(m_aug1, one_hot_tar, reduction='batchmean'))
                    p_a2 = 0.5*(F.kl_div(m_aug2, sig_aug2, reduction='batchmean')) + 0.5*(F.kl_div(m_aug2, one_hot_tar, reduction='batchmean'))
                    sim=(p_c**2 + p_a1**2 + p_a2**2)/3                    

                    if config.only_jsd_scale:
                        mean_loss += 12*sim
                    else:
                        #import ipdb;ipdb.set_trace()
                        mean_loss += 12*sim + loss
                loss = mean_loss

            #torch.stack(sim).mean(dim=1).mean()*loss            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()
            scheduler.step()
            loss_ema = loss_ema * 0.9 + float(loss) * 0.1
            tepoch.set_description(f"Epoch {epoch}")
            tepoch.set_postfix(train_loss=loss_ema)
            if config.disable_tqdm and iter % config.print_freq == 0:
                logging.info('Train Loss {:.3f}'.format(loss))
            iter +=1

    return loss_ema