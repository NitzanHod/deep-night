def sprint(tensor, name=None):
    params = {}
    res = name if name else ''
    params['min'] = "%.2f" % float(tensor.min())
    params['mean'] = "%.2f" % float(tensor.mean())
    params['median'] = "%.2f" % float(tensor.median())
    params['max'] = "%.2f" % float(tensor.max())
    params['shape'] = tensor.shape
    for k, v in params.items():
        res += ' || ' + str(k) + ': ' + str(v)
    print(res)


def _to_camel_case(in_str):
    components = in_str.split('_')
    return ''.join(x.title() for x in components)
