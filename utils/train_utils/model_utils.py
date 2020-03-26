import torch

def initialize_weights(model, path):
    model_dict = model.state_dict()
    loaded_state_dict = torch.load(path, map_location='cuda:0')
    for model_key, loaded_value in zip(model_dict.keys(), loaded_state_dict.values()):
        model_dict[model_key] = loaded_value
    model.load_state_dict(model_dict)
    return model
