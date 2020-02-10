# Experiment management class, helps injecting sacred configs easily
from sacred import Experiment, Ingredient
import yaml


def _to_camel_case(in_str):
    components = in_str.split('_')
    return ''.join(x.title() for x in components)


class ExperimentManager:

    def __init__(self, config_type: str):
        self.component_list = []
        self.config_type = config_type
        self.config = self.setup_cfg()

    def setup_cfg(self):
        if self.config_type in ['sid', 'deepisp', 'test']:
            path = f'cfg/full_cfg_{self.config_type}.yml'
            with open(path) as f:
                return yaml.safe_load(f)
        else:
            raise ValueError('Unexpected experiment type! View cfg_utils.py!')

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
