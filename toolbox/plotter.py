
'''
Plotter class for plotting various things
'''
import os
import shutil
import sys
import copy
import itertools
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


from PIL import Image
import numpy as np

try:
    from visdom import Visdom
except ImportError:
    Visdom = None

try:
    import seaborn as sns
except ImportError:
    sns = None


'''
             .                 .    o8o
           .o8               .o8    `"'
 .oooo.o .o888oo  .oooo.   .o888oo oooo   .ooooo.
d88(  "8   888   `P  )88b    888   `888  d88' `"Y8
`"Y88b.    888    .oP"888    888    888  888
o.  )88b   888 . d8(  888    888 .  888  888   .o8
8""888P'   "888" `Y888""8o   "888" o888o `Y8bod8P'
'''


# get plot data from logger and plot to image file
def save_plot(args, logger, tags=['train', 'val'], name='loss', title='loss curves', labels=None):
    var_dict = copy.copy(logger.logged)
    labels = tags if labels is None else labels

    epochs = None
    for tag in tags:
        if epochs is None:
            epochs = np.array([x for x in var_dict[tag][name].keys()]) 

        curr_line = np.array([x for x in var_dict[tag][name].values()])
        plt.plot(epochs, curr_line)


    plt.grid(True)
    plt.xlabel('epochs')
    plt.title('{} - {}'.format(title, args.name))
    plt.legend(labels=labels)

    out_fn = os.path.join(args.log_dir, 'pics', '{}_{}.png'.format(args.name, name))
    plt.savefig(out_fn, bbox_inches='tight', dpi=150)
    plt.gcf().clear()
    plt.close()


def save_as_best(is_best, out_fn, extension='png'):
    if is_best:
        shutil.copyfile(out_fn, out_fn.replace(f'.{extension}', f'_best.{extension}'))


def plot_confusion_matrix(cm, classnames, out_fn,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    # classnames =  np.array(classnames)    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig, ax = plt.subplots()
    plt.imshow(cm, interpolation='nearest', cmap=cmap, shape=(15000,15000),extent=[0, len(classnames), len(classnames), 0])
    plt.title(title)
    plt.colorbar()
    plt.grid(True)
    tick_marks = np.arange(len(classnames))
    # plt.xticks(tick_marks, classnames, rotation=90)
    # plt.yticks(tick_marks, classnames)

    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classnames, rotation=90, fontsize=5)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classnames, fontsize=5)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    print(f'saving plot to {out_fn}')
    plt.savefig(out_fn, bbox_inches='tight', dpi=300)
    plt.gcf().clear()
    plt.close()    
