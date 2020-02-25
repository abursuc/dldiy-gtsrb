'''
This is the main script of the repository used for launching experiments.
It aggregates input arguments and decides upon them what tasks to perform.
'''

import os
import sys
import shutil

import torch

from loaders import get_loader
from models import get_model
from toolbox import utils, logger, metrics, losses, optimizers
import trainer as trainer
from args import parse_args

from torch.utils.tensorboard import SummaryWriter



'''
 o8o               o8o      .
 `"'               `"'    .o8
oooo  ooo. .oo.   oooo  .o888oo
`888  `888P"Y88b  `888    888
 888   888   888   888    888
 888   888   888   888    888 .
o888o o888o o888o o888o   "888"
'''

def init_logger(args, model):
    # set loggers
    exp_name = args.name
    exp_logger = logger.Experiment(exp_name, args.__dict__)
    exp_logger.add_meters('train', metrics.make_meters(args.num_classes))
    exp_logger.add_meters('val', metrics.make_meters(args.num_classes))
    exp_logger.add_meters('hyperparams', {'learning_rate': metrics.ValueMeter()})
    return exp_logger


'''
          oooo                       .
          `888                     .o8
 .ooooo.   888  oooo  oo.ooooo.  .o888oo  .oooo.o
d88' `"Y8  888 .8P'    888' `88b   888   d88(  "8
888        888888.     888   888   888   `"Y88b.
888   .o8  888 `88b.   888   888   888 . o.  )88b
`Y8bod8P' o888o o888o  888bod8P'   "888" 8""888P'
                       888
                      o888o
'''
def save_checkpoint(args, state, is_best, filename='checkpoint.pth.tar'):
    utils.check_dir(args.log_dir)
    filename = os.path.join(args.log_dir, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(args.log_dir, 'model_best.pth.tar'))

    fn = os.path.join(args.log_dir, 'checkpoint_epoch{}.pth.tar')
    torch.save(state, fn.format(state['epoch']))

    if (state['epoch'] - 1 ) % 5 != 0:
      #remove intermediate saved models, e.g. non-modulo 5 ones
      if os.path.exists(fn.format(state['epoch'] - 1 )):
          os.remove(fn.format(state['epoch'] - 1 ))

    path_logger = os.path.join(args.log_dir, 'logger.json')
    state['exp_logger'].to_json(path_logger)




def load_checkpoint(args, model):
    
    filename = ''

    if 'latest' == args.resume:
        filename = os.path.join(args.log_dir, 'checkpoint.pth.tar')
    elif 'best' == args.resume:
        filename = os.path.join(args.log_dir, 'model_best.pth.tar')
    else:
        filename = os.path.join(args.log_dir, 'checkpoint_epoch{}.pth.tar'.format(args.resume))

    print('Verifying if resume file exists')
    if os.path.exists(filename):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        best_score = checkpoint['best_score']
        best_epoch = checkpoint['best_epoch']
        exp_logger = checkpoint['exp_logger']
        learning_rate = exp_logger.meters['hyperparams']['learning_rate'].val
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
            .format(filename, checkpoint['epoch']))

        return model, exp_logger, start_epoch, best_score, best_epoch, learning_rate
    else:
        print('checkpoint file {} does not exist!'.format(filename))
        return None


'''
                             o8o
                             `"'
ooo. .oo.  .oo.    .oooo.   oooo  ooo. .oo.
`888P"Y88bP"Y88b  `P  )88b  `888  `888P"Y88b
 888   888   888   .oP"888   888   888   888
 888   888   888  d8(  888   888   888   888
o888o o888o o888o `Y888""8o o888o o888o o888o
'''


def main():
    global args, best_score, best_epoch
    best_score, best_epoch = -1, -1
    if len(sys.argv) > 1:
        args = parse_args()
        print('----- Experiments parameters -----')
        for k, v in args.__dict__.items():
            print(k, ':', v)
    else:
        print('Please provide some parameters for the current experiment. Check-out arg.py for more info!')
        sys.exit()    
     
    # init random seeds 
    utils.setup_env(args)

    # init tensorboard summary is asked
    tb_writer = SummaryWriter(f'{args.data_dir}/runs/{args.name}/tensorboard') if args.tensorboard else None

    # init data loaders
    loader = get_loader(args)
    train_loader = torch.utils.data.DataLoader(loader(data_dir=args.data_dir,split='train', 
        phase='train', num_classes=args.num_classes, crop_size=args.crop_size), 
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(loader(data_dir=args.data_dir, split='val',
        phase='test', out_name=True, num_classes=args.num_classes), batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=True)

    exp_logger, lr = None, None

    model = get_model(args)
    criterion = losses.get_criterion(args)

    # optionally resume from a checkpoint
    if args.resume:
        model, exp_logger, args.start_epoch, best_score, best_epoch, lr = load_checkpoint(args, model)
        args.lr = lr
    else:
        # create all output folders 
        utils.init_output_env(args)

    if exp_logger is None:
        exp_logger = init_logger(args, model)

    optimizer, scheduler = optimizers.get_optimizer(args, model)
   
    print('  + Number of params: {}'.format(utils.count_params(model)))

    model.to(args.device)
    criterion.to(args.device)

    if args.test:
        test_loader = torch.utils.data.DataLoader(loader(data_dir=args.data_dir, split='test', 
            phase='test', out_name=True, num_classes=args.num_classes), batch_size=args.batch_size,
            shuffle=False, num_workers=args.workers, pin_memory=True)
        trainer.test(args, test_loader, model, criterion, args.start_epoch, 
            eval_score=metrics.accuracy_classif, output_dir=args.out_pred_dir, has_gt=True)
        sys.exit()

    is_best = True
    for epoch in range(args.start_epoch, args.epochs + 1):
        print('Current epoch: ', epoch)
       
        trainer.train(args, train_loader, model, criterion, optimizer, exp_logger, epoch,
              eval_score=metrics.accuracy_classif, tb_writer=tb_writer)
         
        # evaluate on validation set
        mAP, val_loss, res_list = trainer.validate(args, val_loader, model, criterion, exp_logger, epoch, eval_score=metrics.accuracy_classif, tb_writer=tb_writer)

        # update learning rate
        if scheduler is None:
            trainer.adjust_learning_rate(args, optimizer, epoch)
        else:
            prev_lr =  optimizer.param_groups[0]['lr']
            if 'ReduceLROnPlateau' == args.scheduler:
                scheduler.step(val_loss)
            else:    
                scheduler.step()
                
            print(f"Updating learning rate from {prev_lr} to {optimizer.param_groups[0]['lr']}")


        # remember best acc and save checkpoint
        is_best = mAP > best_score
        best_score = max(mAP, best_score)
        if True == is_best:
            best_epoch = epoch

        save_checkpoint(args, {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_score': best_score,
            'best_epoch': best_epoch,
            'exp_logger': exp_logger,
            'res_list': res_list,
        }, is_best)


    if args.tensorboard:
        tb_writer.close()

    print("That's all folks!")



if __name__ == '__main__':
    main()
