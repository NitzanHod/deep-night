# Experiment management class, helps injecting sacred configs easily
from sacred import Ingredient
import yaml
import os
from shutil import copy2


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class ExperimentManager(metaclass=Singleton):

    def __init__(self, path: str = ''):
        self.config_path, self.config = self.setup_cfg(path)


    def setup_cfg(self, path):
        if path in ['sid', 'testing', 'finetune']:
            path = f'cfg/train_{path}.yml'

        with open(path, 'r') as f:
            return path, yaml.safe_load(f)

    def store_cfg(self, dirname):
        _, filename = os.path.split(self.config_path)
        cfg_store_path = os.path.join(dirname, filename)
        print('Copying cfg file: ', self.config_path, 'to', cfg_store_path, '.')
        copy2(self.config_path, cfg_store_path)  # copy file

    @staticmethod
    def generate_cfg(params: dict, name: str):
        path = 'cfg/train_sid.yml'
        result_path = f'cfg/train_sid_{name}.yml'
        with open(path, 'r') as f:
            dic = yaml.safe_load(f)
        for cfg_section in params.keys():
            for k, v in params[cfg_section].items():
                if k in dic[cfg_section].keys():
                    dic[cfg_section][k] = v
        with open(result_path, 'w') as f:
            yaml.safe_dump(dic, f)

    def get_ingredient(self, name):
        ingredient = Ingredient(name)
        ingredient.add_config(self.config)
        return ingredient


if __name__ == '__main__':
    ExperimentManager.generate_cfg({'train_cfg': {'model': 'cfg/model_cfg/dropout_u_net.cfg'}}, 'dropout')
    ExperimentManager.generate_cfg({'train_cfg': {'weight_decay': 0.1}}, 'reg')
    ExperimentManager.generate_cfg({'train_cfg': {'model': 'cfg/model_cfg/dropout_u_net.cfg', 'weight_decay': 0.1}},
                                   'dropout_reg')
