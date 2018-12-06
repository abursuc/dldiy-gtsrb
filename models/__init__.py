from models.resnet import resnet
from models.squeezenet import squeezenet
from models.lenet import lenet


def get_model(args):
    model_instance = _get_model_instance(args.arch)

    print('Fetching model %s - %s ' % (args.arch, args.model_name))
    if args.arch == 'resnet':
        model = model_instance(args.model_name, args.num_classes, args.input_channels, args.pretrained)
    elif args.arch == 'squeezenet':
        model = model_instance(args.model_name, args.num_classes, args.input_channels, args.pretrained)
    elif args.arch == 'lenet':
        model = model_instance(args.model_name, args.num_classes, args.input_channels)
    else:
        raise 'Model {} not available'.format(args.arch)

    return model

def _get_model_instance(name):
    return {
        'resnet': resnet,
        'squeezenet': squeezenet,
        'lenet': lenet,
    }[name]