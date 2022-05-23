import numpy as np
import torch
from config import datasets, cli
from models import *
import lp.db_eval as db_eval
import torch.backends.cudnn as cudnn
import helpers
import os
from sklearn.metrics import confusion_matrix

import matplotlib
matplotlib.use('Agg')
import itertools
import numpy as np
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, classes, result_path, correct, total, acc):    
    # Unnormalized
    u_cm = cm
    # print('Confusion matrix, without normalization')    
    # print(cm)
    # Normalized confusion matrix
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # print("Normalized confusion matrix")
    # print(cm)    
    
    plt.figure(figsize=(12,9))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.YlGnBu)
    accuracy = '\nAccuracy: {}/{} ({:.5f})\n'.format(correct, total, acc)        
    title = 'Semisupervised 304, label size = 50%, other settings = default. ' + accuracy 
    # title = 'Full supervised 304 (baseline), label size = 50%, other settings = default. ' + accuracy 
            
    plt.title(title, fontsize=16, color="black")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, horizontalalignment="right", rotation=45)
    plt.yticks(tick_marks, classes)

    for i in range(len(classes)):
        plt.axhline(tick_marks[i]+0.5, color='gray', linewidth=0.5)
        plt.axvline(tick_marks[i]+0.5, color='gray', linewidth=0.5)
            
    thresh = cm.max() / 2.
    float_formatter = "{:.1f}".format
    int_formatter = "{:d}".format
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if cm[i,j] != 0:
            normalized_value = float_formatter(cm[i,j]*100)
            unnormalized_value = int_formatter(u_cm[i,j])      
            plt.text(j, i, '{}%\n {}'.format(normalized_value, unnormalized_value), \
            verticalalignment="center", horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
                    
    plt.tight_layout()
    ax = plt.gca()    
    ax.set_xlabel('Predicted Class', fontsize=14, color="cornflowerblue")
    ax.xaxis.set_label_coords(1.07, -0.1)
    plt.ylabel('Actual Class', fontsize=14, color="cornflowerblue")
    # plt.xlabel('Predicted class', fontsize=14)    
    savepath = result_path + "/semi_label60percent_exp2.png"
    plt.savefig(savepath)

# --------------

def create_model(num_classes,args):
    model = resnet10(num_classes) 
    args.device = torch.device('cuda')   
    device_string = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(args.device)
    model.load_state_dict(torch.load(
        args.weights_path, map_location=device_string))
    cudnn.benchmark = True    
    return model


def validate(eval_loader, model, args, num_classes =15):
    predictions = []
    correct = 0   
    y_true = []
    y_pred = [] 
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(eval_loader):
            batch_size = targets.size(0)
            model.eval()
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs,_ = model(inputs)
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            
            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            
            pred = outputs.max(1, keepdim=True)[1]                        
            pred_eq_target = pred.eq(targets.view_as(pred)).squeeze(1).cpu().numpy()
            predictions += list(pred.squeeze(1).cpu().numpy())
            correct += pred_eq_target.sum()
            total += batch_size
            
            # print("pred, target: {}, {}".format(pred[0].item(), targets[0].item()))
            for i in range(batch_size):
                y_true.append(targets[i].item())
                y_pred.append(pred[i].item())
    
        cm = confusion_matrix(y_true, y_pred)
        print(cm)
    return cm, correct, total, correct/total
            
            
     
           
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    # print(pred.shape)

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def eval(weak_transformation,strong_transformation,
                        eval_transformation,
                        datadir,
                        args):
    evaldir = os.path.join(datadir, args.eval_subdir)
    eval_dataset = db_eval.DBE(evaldir, False, eval_transformation)
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
        drop_last=False)
    
    return eval_loader

def eval_main():
    #### Get the command line arguments
    args = cli.parse_commandline_args()
    args = helpers.load_args(args)
    dataset_config = datasets.__dict__[args.dataset]()
    num_classes = dataset_config.pop('num_classes')
    model = create_model(num_classes,args)    

    eval_loader = eval(**dataset_config, args=args)  
    
    #### Plot confusion matrix
    classes_names = ['ROUND-DEFAULT-PINK', 'ROUND-DEFAULT-BLUE', 'ROUND-DEFAULT-VIOLET',\
        'ROUND-DEFAULT-RED', 'ROUND-DEFAULT-LIGHT_BLUE', 'ROUND-DEFAULT-GREEN', \
        'ROUND-DEFAULT-BLACK',' ROUND-DEFAULT-ORANGE', 'ROUND-DEFAULT-YELLOW', \
        'ROUND-DEFAULT-DARK_BLUE', 'ROUND-NN-RED', 'ROUND-NN-GREEN', \
        'ROUND-NN-BLACK', 'ROUND-NN-ORANGE', 'ROUND-NN-YELLOW']
    cm, correct, total, acc = validate(eval_loader, model, args, num_classes)
    plot_confusion_matrix(cm, classes_names, args.logdir, correct, total, acc)



if __name__ == '__main__':
    eval_main()
