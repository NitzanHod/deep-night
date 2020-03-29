import torch
import numpy as np

def sprint(tensor, name=None):
    params = {}
    res = name if name else ''
    params['min'] = "%.2f" % float(tensor.min())
    params['mean'] = "%.2f" % float(tensor.mean())

    if type(tensor) == torch.Tensor:
        params['median'] = "%.2f" % float(tensor.median())
    elif type(tensor) == np.ndarray:
        params['median'] = "%.2f" % float(np.median(tensor))

    params['max'] = "%.2f" % float(tensor.max())
    params['shape'] = tensor.shape
    for k, v in params.items():
        res += ' || ' + str(k) + ': ' + str(v)
    print(res)


def _to_camel_case(in_str):
    components = in_str.split('_')
    return ''.join(x.title() for x in components)


import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_tensorflow_log(path):

    # Loading too much data is slow...
    tf_size_guidance = {
        'compressedHistograms': 10,
        'images': 0,
        'scalars': 100,
        'histograms': 1
    }

    event_acc = EventAccumulator(path, tf_size_guidance)
    event_acc.Reload()

    # Show all tags in the log file
    #print(event_acc.Tags())

    training_accuracies =  event_acc.Scalars('training-accuracy')
    validation_accuracies = event_acc.Scalars('validation_accuracy')

    steps = 10
    x = np.arange(steps)
    y = np.zeros([steps, 2])

    for i in range(steps):
        y[i, 0] = training_accuracies[i][2] # value
        y[i, 1] = validation_accuracies[i][2]

    plt.plot(x, y[:,0], label='training accuracy')
    plt.plot(x, y[:,1], label='validation accuracy')

    plt.xlabel("Steps")
    plt.ylabel("Accuracy")
    plt.title("Training Progress")
    plt.legend(loc='upper right', frameon=True)
    plt.show()


if __name__ == '__main__':
    log_file = "./logs/events.out.tfevents.1456909092.DTA16004"
    plot_tensorflow_log(log_file)
