import logging
logger = logging.getLogger('base')


def create_model(opt):
    model = opt['model']

    if model == 'sr':
        # from .SR_model import SRModel as M
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    elif model == 'srgan':
        from .SRGAN_model import SRGANModel as M
        # raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    elif model == 'srragan':
        from .SRRaGAN_model import SRRaGANModel as M
    elif model == 'sftgan':
        # from .SFTGAN_ACD_model import SFTGAN_ACD_Model as M
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
