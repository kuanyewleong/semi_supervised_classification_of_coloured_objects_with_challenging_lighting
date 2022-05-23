# Coloured Objects Classification under challenging Lighting Conditions
## (semisupervised learning with a Hybrid Energy-Neural-Graphical approach)

Technique based on the paper: https://arxiv.org/pdf/2106.04527.pdf, but several parts are optimized / modified to suit challenging colour classification scenario. Major adaptation for colour augmentation used in semi-supervised learning is given in the lasst section below.

You may need to do your own calculation for pixel statistics and replace your values in "datasets.py". You can refer the formulae in "calculate_image_mean_std.py"

Currently it is set as the following for my use case (private dataset):
mean: tensor([0.0396, 0.0273, 0.0230])
std:  tensor([0.0917, 0.0671, 0.0535])

Example of running the scrip for (--dataset star refers to my private ddatset, you can modify yours in datasets.py):
```bash
python3 main.py --dataset star --num-labeled 7680 --alpha 1.0 --lr 0.03 --labeled-batch-size 128 --batch-size 512 --aug-num 3 --label-split 1 --progress True --num-steps 250000
```

Example of testing and ploting confusion matrix:
```bash
python3 eval_plot_cm_default.py --dataset star --weights_path runs/state_dict_epoch_100.pth --num-labeled 33600 --label-split 1 --logdir runs_eval
```
(For the testing script, I am just reusing the training interface, it can be simplified later when time permit.)

## Command line arguments

The documentation for the command line arguments can be found in config/cli.py. Here are some extra information on the most important ones.

- --dataset : If you want to add other dataset you would need to update the config/datasets.py folder to include your new dataset and then finally change the load_args function in the helpers.py. You may potentially need to change the --train-subdir and --eval-subdir options as well to make sure you are pointing to the right folders. 

- --model : Current avaiable option is our default resnet10. If you want to add your own custom model you would need to add the code to the models subfolder, update the init and then add your model as an option to the create-model function in helpers.py

- --label-split : For better observation, control and comparison of experiments, I prepared random split of balancced labels. These are numbered from 1 to 3 for each differing label amount. This label split is then used in the create_data_loaders_simple function in helpers.py where it is sent to the custom dataset class db_semisuper. For some data sets it may make more sense to generate such splits on the fly by changing the relabel dataset function of the db_semisuper class found in the lp folder.

- --aug-num : This sets the number of augmentation samples per point as dicussed in the main paper. A value of 3 or 5 is best in most cases.

## Balanced Labels
For research or training purpose, some pre-generated balanced class labels are given in data-local/labels/. 
I ran 3 exepriments for every case, so 3 split of different random samples are given in the repository to cater for them, i.e.:
The structure in 304 for 20 percent balanced labels are: 

data-local/labels/r304/7590_balanced_labels_20percent/  |01.txt  |02.txt  |03.txt

The number label "7590" in the directory "7590_balanced_labels" is also the amount equvalent to the 20 percent of the total training samples.

To run with the 20 percent samples of balanced labels, use the argument --num-labeled 7590

To run with the split of random samples in 01.txt, use the argument --label-split 1

## Data Structure
In this analysis, we sort the train data as the following structure for the time being for the data to be loaded.  
        
        root/class_x/imagex.png
        root/class_x/imagey.png
        root/class_x/imagez.png

        root/class_y/imagex.png
        root/class_y/imagey.png
        root/class_y/imagez.png

So before running any training, make sure to prepare you dataset accordingly.


## PCA augmentation is the major augmentation to achieve better colour augmentation used in this research
You can take a look at this function in config/augmentations.py
(I am currently experimentating using SVD for faster calculation, update soon...)

```python
def pca_color_augmentation(image_array_input):
    assert image_array_input.ndim == 3 and image_array_input.shape[2] == 3
    assert image_array_input.dtype == np.uint8

    img = image_array_input.reshape(-1, 3).astype(np.float32)
    img = (img - np.mean(img, axis=0)) / np.std(img, axis=0)

    cov = np.cov(img, rowvar=False)
    lambd_eigen_value, p_eigen_vector = np.linalg.eig(cov)

    rand = np.random.randn(3) * 0.1
    delta = np.dot(p_eigen_vector, rand*lambd_eigen_value)
    delta = (delta * 255.0).astype(np.int32)[np.newaxis, np.newaxis, :]

    img_out = np.clip(image_array_input + delta, 0, 255).astype(np.uint8)
    return img_out
```
