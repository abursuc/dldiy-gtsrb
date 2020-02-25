from loaders.gtsrb_loader import GTSRBLoader

def get_loader(args):
    """get_loader

    :param name:
    """
    return {
        'gtsrb' : GTSRBLoader,
        # feel free to add new datasets here
    }[args.dataset]
