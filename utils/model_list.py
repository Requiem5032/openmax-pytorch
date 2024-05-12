from utils.xception import Mos_Xception


def xception(config):
    return Mos_Xception(config=config, model_name='xception', num_classes=config.num_species.sum())
