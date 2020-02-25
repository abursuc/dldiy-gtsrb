# pytorch-gtsrb

This is an illustrative example for a deep learning project in pytorch for the [DL-DIY course](https://mlelarge.github.io/dataflowr-web/dldiy.html). We use the [German Traffic Sign Recognition Benchmark (GTSRB)](http://benchmark.ini.rub.de/?section=gtsrb&subsection=news) to show how could the code be structured for easier testing of new architectures and parameters, tracking of results and improving of the models.

## Data
The original GTSRB datasets consists of large `.ppm` images of scenes with bounding box coordinates for the traffic signs. We use here a post-processed variant where signs have already been cropped out from their corresponding images and resized to 32 x 32. 

We use a similar set-up with the Udacity course for the data. You can download the zipped data from [here](https://drive.google.com/open?id=1E3cc8yNBL5vZ8XGVOKuWY9w8BpO4EVbK).

A presentation of the dataset, its content and statistics on classes can be accessed [here](https://github.com/mohamedameen93/German-Traffic-Sign-Classification-Using-TensorFlow).

## Project structure

The project is structured as following:

```bash
.
├── loaders
|   └── dataset selector
|   └── gtsrb_loader.py # loading and pre-processing gtsrb data
├── models
|   └── architecture selector
|   └── lenet.py # classical CNN
|   └── resnet.py # relatively recent CNN 
|   └── squeezenet.py # highly compressed CNN
├── toolbox
|   └── optimizer and losses selectors
|   └── logger.py  # keeping track of most results during training and storage to static .html file
|   └── metrics.py # code snippets for computing scores and main values to track
|   └── plotter.py # snippets for plotting and saving plots to disk
|   └── utils.py   # various utility functions
├── commander.py # main file from the project serving for calling all necessary functions for training and testing
├── args.py # parsing all command line arguments for experiments
├── trainer.py # pipelines for training, validation and testing
```

## Launching
Experiments can be launched by calling `commander.py` and a set of input arguments to customize the experiments. You can find the list of available arguments in `args.py` and some default values. Note that not all parameters are mandatory for launching and most of them will be assigned their default value if the user does not modify them.

Here is a typical launch command and some comments:

- `python commander.py --dataset gtsrb --name gtsrb_lenet_optsgd_lr1e-3_lrdecayPlateau0.5_bsz128 --batch-size 128 --optimizer sgd --scheduler ReduceLROnPlateau --lr 1e-3 --lr-decay 0.5 --step 15 --epochs 100 --arch lenet --model-name lenet5 --root-dir /data/ --num-classes 43 --workers 4 --crop-size 32 --criterion crossentropy --tensorboard`
  + this experiment is on the _gtsrb_ dataset which can be found in `--root-dir/gtsrb` trained over _LeNet5_. It optimizes with _sgd_ with initial learning rate (`--lr`) of `1e-3` which is decayed by half whenever the `--scheduler` _ReduceLRonPlateau_ does not see an improvement in the validation accuracy for more than `--step` epochs. Input images are of size 32  (`--crop-size`). In addition it saves intermediate results to `--tensorboard`.
  + when debugging you can add the `--short-run` argument which performs training and validation epochs of 10 mini-batches. This allows testing your entire pipeline before launching an experiment
  + if you want to resume a previously paused experiment you can use the `--resume` flag which can continue the training from _best_, _latest_ or a specifically designated epoch.
  + if you want to use your model only for evaluation on the test set, add the `--test` flag.
  + when using _resnet_ you should upsample the input images to ensure compatibility. To this end set `--crop-size 64`
 
## Output
For each experiment a folder with the same name is created in the folder `root-dir/gtsrb/runs`
 This folder contains the following items:

```bash
.
├── checkpoints (\*.pth.tar) # models and logs are saved every epoch in .tar files. Non-modulo 5 epochs are then deleted.
├── best model (\*.pth.tar) # the currently best model for the experiment is saved separately
├── config.json  # experiment hyperparameters
├── logger.json  # scores and metrics from all training epochs (loss, learning rate, accuracy,etc.)
├── res  # predictions for each sample from the validation set for every epoch
├── tensorboard  # experiment values saved in tensorboard format
 ```

### Tensorboard
In order the visualize metrics and results in tensorboard you need to launch it separately: `tensorboard --logdir /root-dir/gtsrb/runs`. You can then access tensorboard in our browser at [localhost:6006](localhost:6006)
If you have performed multiple experiments, tensorboard will aggregate them in the same dashboard.
  
  
 ## Dependencies
 - [imageio](http://imageio.github.io/)
