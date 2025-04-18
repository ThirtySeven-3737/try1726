import os
import argparse
from utils import Storage

class Configs():
    def __init__(self, args):
        # hyper parameters for models
        HYPER_MODEL_MAP = {
            'mobilenetv3_large': self.__mobilenetv3_large
        }

        # normalize
        model_name = str.lower(args.modelName)
        dataset_name = str.lower(args.datasetName)
        # load params
        # integrate all parameters
        self.args = Storage(dict(vars(args),
                            **HYPER_MODEL_MAP[model_name]()['datasetParas'][dataset_name],
                            ))

    def __mobilenetv3_large(self):
        tmp = {
            # dataset
            'datasetParas':{
                'dreamer':{
                    # the batch_size of each epoch is update_epochs * batch_size
                    'batch_size': 64,
                    'learning_rate': 1e-3,       
                    'early_stop': 8,
                    'input_channels':2, 
                    'num_classes': 5,
                    'epoches': 5
                },
                'amigos':{
                    'batch_size': 64,
                    'learning_rate': 1e-3,
                    'early_stop': 8,
                    'input_channels':2, 
                    'num_classes': 9,
                    'epoches': 50
                },
            },
        }
        return tmp

    def get_config(self):
        return self.args

