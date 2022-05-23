# color-recognition (semisupervised learning with a Hybrid Energy-Neural-Graphical approach)

Technique based on the paper: https://arxiv.org/pdf/2106.04527.pdf, but several parts are optimized / modified to suit our CR scenario.

You may use the following pixel statistics for CR dataset in "datasets.py", but you may need to recalculate if other datset is used.

For theStar:
mean: tensor([0.0396, 0.0273, 0.0230])
std:  tensor([0.0917, 0.0671, 0.0535])

For Room 304:
mean: tensor([0.0536, 0.0469, 0.0472])
std:  tensor([0.1033, 0.0853, 0.0898])

For TheStar Area3 only:
mean: tensor([0.0359, 0.0248, 0.0214])
std:  tensor([0.0648, 0.0414, 0.0293])

For TheStar Area5 only:
mean: tensor([0.0161, 0.0142, 0.0145])
std:  tensor([0.0135, 0.0091, 0.0105])


Example of running the scrip for TheStar dataset:
```bash
python3 main.py --dataset star --num-labeled 7680 --alpha 1.0 --lr 0.03 --labeled-batch-size 128 --batch-size 512 --aug-num 3 --label-split 1 --progress True --num-steps 250000
```

Example of testing and ploting confusion matrix for 304
```bash
python3 eval_plot_cm_default.py --dataset r304 --weights_path runs/semisupervised/304/18945_balanced_labels_50percent/state_dict_epoch_100.pth --num-labeled 33600 --label-split 1 --logdir runs_eval
```
(For the testing script, I am just reusing the training interface, it can be simplified later when time permit.)

## Command line arguments

The documentation for the command line arguments can be found in config/cli.py. Here are some extra information on the most important ones.

- --dataset : Current available options are star, r304, area3, area5. If you want to add other dataset you would need to update the config/datasets.py folder to include your new dataset and then finally change the load_args function in the helpers.py. You may potentially need to change the --train-subdir and --eval-subdir options as well to make sure you are pointing to the right folders. 

- --model : Current avaiable option is our default resnet10 (without LogSobel). If you want to add your own custom model you would need to add the code to the models subfolder, update the init and then add your model as an option to the create-model function in helpers.py

- --label-split : For better observation, control and comparison of experiments, I prepared random split of balancced labels. These are numbered from 1 to 3 for each differing label amount. This label split is then used in the create_data_loaders_simple function in helpers.py where it is sent to the custom dataset class db_semisuper. For some data sets it may make more sense to generate such splits on the fly by changing the relabel dataset function of the db_semisuper class found in the lp folder.

- --aug-num : This sets the number of augmentation samples per point as dicussed in the main paper. A value of 3 or 5 is best in most cases.

## Balanced Labels
For research or training purpose, some pre-generated balanced class labels of the Star, 304, Area3 samples are given in data-local/labels/. 
We run 3 exepriments for every case, so 3 split of different random samples are given in the repository to cater for them, i.e.:
The structure in 304 for 20 percent balanced labels are: 

data-local/labels/r304/7590_balanced_labels_20percent/  |01.txt  |02.txt  |03.txt

The number label "7590" in the directory "7590_balanced_labels_20percent" is also the amount equvalent to the 20 percent of the total training samples.

To run with the 20 percent samples of balanced labels, use the argument --num-labeled 7590

To run with the split of random samples in 01.txt, use the argument --label-split 1

Labels of large percentage like 50% is too large to be shared here, so you can use the "data-local/labels/generate_balanced_labels.py" to generate yourr own balanced labels if needed. 

## Data Structure
In this analysis, we sort the train data as the following structure for the time being for the data to be loaded. This is conveniently a temporary solution to avoid CPU out of memory issue due to the way our default CR data loader. It allows for not only the full size of large data like TheStar to be loaded for training, also it is not affecting the extra image augmentation needed for semisupervised training. 
        
        root/class_x/imagex.png
        root/class_x/imagey.png
        root/class_x/imagez.png

        root/class_y/imagex.png
        root/class_y/imagey.png
        root/class_y/imagez.png

So before running any training, make sure to prepare you dataset accordingly.

A simple script like this could help generate the needed subfolder for TheStar CR dataset:
```python
import os

target_names = ['ROUND-DEFAULT-PINK', 'ROUND-DEFAULT-BLUE', 'ROUND-DEFAULT-VIOLET',\
        'ROUND-DEFAULT-RED', 'ROUND-DEFAULT-LIGHT_BLUE', 'ROUND-DEFAULT-GREEN', \
        'ROUND-DEFAULT-BLACK',' ROUND-DEFAULT-ORANGE', 'ROUND-DEFAULT-YELLOW', \
        'ROUND-DEFAULT-DARK_BLUE', 'ROUND-NN-RED', 'ROUND-NN-GREEN', \
        'ROUND-NN-BLACK', 'ROUND-NN-ORANGE', 'ROUND-NN-YELLOW']

for i in range(len(target_names)):
    command = "mkdir " + target_names[i]
    os.system(command)
```
Please let me know if you need the readily available dataset for experiment, I can share with you the dataset I prepared in Weil.