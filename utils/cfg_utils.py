# Experiment management class, helps injecting sacred configs easily
from sacred import Experiment, Ingredient
import yaml
import os
from shutil import copy2


def _to_camel_case(in_str):
    components = in_str.split('_')
    return ''.join(x.title() for x in components)


class ExperimentManager:

    def __init__(self, path: str):
        self.component_list = []
        self.config_path, self.config = self.setup_cfg(path)

    def setup_cfg(self, path):
        if path in ['sid', 'deepisp', 'testing']:
            path = f'cfg/train_{path}.yml'

        with open(path, 'r') as f:
            return path, yaml.safe_load(f)

    def append(self, name: str):
        new_component = Component(name, self.config)
        self.component_list.append(new_component)
        methods = new_component.methods
        for method in methods:
            setattr(self, method.__name__, method)

    def prepare_run(self, component_names):
        for component_name in component_names:
            self.append(component_name)
        ingredients = [comp.module_ingredient for comp in self.component_list]
        ex = Experiment(ingredients=ingredients)
        ex.add_config(self.config)
        return ex

    def store_cfg(self, dirname):
        _, filename = os.path.split(self.config_path)
        cfg_store_path = os.path.join(dirname, filename)
        copy2(self.config_path, cfg_store_path)  # copy file

    @staticmethod
    def generate_cfg(params : dict , name : str):
        path = 'cfg/train_sid.yml'
        result_path = f'cfg/train_sid_{name}.yml'
        with open(path, 'r') as f:
            dic = yaml.safe_load(f)
        for cfg_section in params.keys():
            for k,v in params[cfg_section].items():
                if k in dic[cfg_section].keys():
                    dic[cfg_section][k] = v
        with open(result_path, 'w') as f:
            yaml.safe_dump(dic, f)

class Component:
    def __init__(self, component_name, config):
        self.component_name = component_name
        self.config = config
        self.module_ingredient = self.prepare_ingredient(component_name)
        self.methods = self.get_methods(self.component_name)

    def prepare_ingredient(self, component_name):
        module_ingredient = Ingredient(component_name)
        module_ingredient.add_config(self.config)
        return module_ingredient

    def get_methods(self, component_name):
        class_name = _to_camel_case(component_name) + 'Component'
        import_path = f'utils.train_utils.{component_name}'
        module = __import__(import_path, fromlist=[class_name])

        component_class = getattr(module, class_name)
        component_instance = component_class(self.module_ingredient)
        return component_instance.methods

    def __str__(self):
        return f'Component[{self.component_name}]'

    def __repr__(self):
        return str(self)


if __name__ == '__main__':
    ExperimentManager.generate_cfg({'train_cfg':{'model':'cfg/model_cfg/dropout_u_net.cfg'}}, 'dropout')
    ExperimentManager.generate_cfg({'train_cfg':{'weight_decay': 0.1}}, 'reg')
    ExperimentManager.generate_cfg({'train_cfg':{'model':'cfg/model_cfg/dropout_u_net.cfg', 'weight_decay': 0.1}}, 'dropout_reg')
