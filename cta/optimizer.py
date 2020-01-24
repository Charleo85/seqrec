import torch.optim as optim

class WarmupConstantSchedule(optim.lr_scheduler.LambdaLR):
    """ Linear warmup and then constant.
        Linearly increases learning rate schedule from 0 to 1 over `warmup_steps` training steps.
        Keeps learning rate schedule equal to 1. after warmup_steps.
    """
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(WarmupConstantSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        return 1.

class WarmupLinearSchedule(optim.lr_scheduler.LambdaLR):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))
    
class WarmupCosineSchedule(optim.lr_scheduler.LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))


class WarmupCosineWithHardRestartsSchedule(optim.lr_scheduler.LambdaLR):
    """ Linear warmup and then cosine cycles with hard restarts.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        If `cycles` (default=1.) is different from default, learning rate follows `cycles` times a cosine decaying
        learning rate (with hard restarts).
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=1., last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineWithHardRestartsSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        if progress >= 1.0:
            return 0.0
        return max(0.0, 0.5 * (1. + math.cos(math.pi * ((float(self.cycles) * progress) % 1.0))))


class Optimizer:
    def __init__(self, params, optimizer_type='Adagrad', lr=.05,
                 momentum=0, weight_decay=0, eps=1e-6):
        '''
        An abstract optimizer class for handling various kinds of optimizers.
        You can specify the optimizer type and related parameters as you want.
        Usage is exactly the same as an instance of torch.optim
        Args:
            params: torch.nn.Parameter. The NN parameters to optimize
            optimizer_type: type of the optimizer to use
            lr: learning rate
            momentum: momentum, if needed
            weight_decay: weight decay, if needed. Equivalent to L2 regulariztion.
            eps: eps parameter, if needed.
        '''
        if optimizer_type == 'RMSProp':
            self.optimizer = optim.RMSprop(params, lr=lr,
                                           eps=eps,
                                           weight_decay=weight_decay,
                                           momentum=momentum)
        elif optimizer_type == 'Adagrad':
            self.optimizer = optim.Adagrad(params, lr=lr,
                                           weight_decay=weight_decay)
        elif optimizer_type == 'Adadelta':
            self.optimizer = optim.Adadelta(params,
                                            lr=lr,
                                            eps=eps,
                                            weight_decay=weight_decay)
        elif optimizer_type == 'Adam':
            self.optimizer = optim.Adam(params,lr=lr,eps=eps,weight_decay=weight_decay)
            
        elif optimizer_type == 'SparseAdam':
            self.optimizer = optim.SparseAdam(params,
                                              lr=lr,
                                              eps=eps)
        elif optimizer_type == 'SGD':
            self.optimizer = optim.SGD(params,lr=lr,momentum=momentum, weight_decay=weight_decay)
        else:
            raise NotImplementedError
            
#         self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[3,6,9,12], gamma=5)
        self.scheduler = WarmupLinearSchedule(self.optimizer,  warmup_steps=4, t_total=12)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()
        
    def step_scheduler(self):
        self.scheduler.step()

