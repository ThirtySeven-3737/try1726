import os
import argparse
from utils import Storage

class Configs():
    def __init__(self, args):
        # hyper parameters for models
        HYPER_MODEL_MAP = {
            'mobilenetv3_small': self.__mobilenetv3_small
        }

        # normalize
        model_name = str.lower(args.modelName)
        dataset_name = str.lower(args.datasetName)
        # load params
        # integrate all parameters
        self.args = Storage(dict(vars(args),
                            **HYPER_MODEL_MAP[model_name]()['datasetParas'][dataset_name],
                            ))

    def __mobilenetv3_small(self):
        tmp = {
            # dataset
            'datasetParas':{
                'dreamer':{
                    # the batch_size of each epoch is update_epochs * batch_size
                    'batch_size': 16,
                    'learning_rate': 1e-3,       
                    'early_stop': 8,
                    'input_channels':2, 
                    'num_classes': 4,
                    'epoches': 50
                },
                'amigos':{
                    'batch_size': 32,
                    'learning_rate': 1e-3,
                    'early_stop': 8,
                    'input_channels':2, 
                    'num_classes': 4,
                    'epoches': 50
                },
                'swell': {
                    'batch_size': 4,
                    'learning_rate': 1e-3,
                    'early_stop': 8,
                    'input_channels': 1,
                    'num_classes': 4,
                    'epoches': 50
                },
            },
        }
        return tmp

    def get_config(self):
        return self.args

