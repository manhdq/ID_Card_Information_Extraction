from torch.optim import lr_scheduler


def fetch_scheduler(opt, optimizer):
    if opt.scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=opt.T_max, 
                                                   eta_min=opt.min_lr)
    elif opt.scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=opt.T_0, 
                                                             eta_min=opt.min_lr)
    elif opt.scheduler == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   mode='min',
                                                   factor=0.1,
                                                   patience=7,
                                                   threshold=0.0001,
                                                   min_lr=opt.min_lr,)
    elif opt.scheduer == 'ExponentialLR':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
    elif opt.scheduler == None:
        return None
        
    return scheduler