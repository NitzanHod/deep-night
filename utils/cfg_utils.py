# Experiment management class, helps injecting sacred configs easily
class ExperimentManager:
    def __init__(self, experiment_type: str):
        self.experiment_type = experiment_type

    def get_cfg(self):
        if self.experiment_type in ['sid', 'deepisp', 'test']:
            return f'cfg/full_cfg_{self.experiment_type}.json'
        else:
            raise ValueError('Unexpected experiment type! View cfg_utils.py!')
