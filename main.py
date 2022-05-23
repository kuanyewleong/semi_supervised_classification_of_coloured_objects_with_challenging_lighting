import time
import numpy as np
import torch
from config import datasets, cli
import helpers
import math
from pathlib import Path
from tensorboardX import SummaryWriter

def laplacenet():
	#### Get the command line arguments
	args = cli.parse_commandline_args()
	args = helpers.load_args(args)
	# args.file = "_" + args.dataset + "_" + str(args.num_labeled) + "_" + str(args.label_split) + "_" + str(args.num_steps) + "_" + str(args.aug_num) +  ".txt"

	if args.seed is not None:
		# RNG control
		random.seed(args.seed)

		# https://pytorch.org/docs/stable/notes/randomness.html
		torch.manual_seed(args.seed)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
	    
	#### Load the dataset
	dataset_config = datasets.__dict__[args.dataset]()
	num_classes = dataset_config.pop('num_classes')
	args.num_classes = num_classes

	#### Create loaders
	#### train_loader loads the labeled data , eval loader is for evaluation
	#### train_loader_noshuff extracts features 
	#### train_loader_l, train_loader_u together create composite batches
	#### dataset is the custom dataset class
	train_loader, eval_loader , train_loader_noshuff , train_loader_l , train_loader_u , dataset = helpers.create_data_loaders_simple(**dataset_config, args=args)
	
	#### Create Model and Optimiser 
	args.device = torch.device('cuda')
	model = helpers.create_model(num_classes,args)
	# optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9,weight_decay=args.weight_decay, nesterov=args.nesterov)
	optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

	#### Transform steps into epochs
	num_steps = args.num_steps
	ini_steps = math.floor(args.num_labeled/args.batch_size)*100
	ssl_steps = math.floor( len(dataset.unlabeled_idx) / ( args.batch_size - args.labeled_batch_size))
	args.epochs = 10 + math.floor((num_steps - ini_steps) / ssl_steps)
	args.lr_rampdown_epochs = args.epochs + 10
	print("Epochs: ", args.epochs)

	#### Information store in epoch results and then saved to file
	global_step = 0
	epoch_results = np.zeros((args.epochs,6))	

	#### Save model params
	writer = SummaryWriter(args.logdir)
	with (Path(args.logdir) / 'params.txt').open('w') as file:
		print('### model ###', file=file)
		print(model, file=file)
		print('### train params ###', file=file)
		print(args, file=file)

	#%%
	supervised_init = 10
	for epoch in range(args.epochs):
		start_epoch_time = time.time()
	    #### Extract features and run label prop on graph laplacian
		if epoch >= supervised_init:
			dataset.feat_mode = True
			feats = helpers.extract_features_simp(train_loader_noshuff,model,args)  
			dataset.feat_mode = False          
			dataset.one_iter_true(feats, writer, epoch, k = args.knn, max_iter = 30, l2 = True , index="ip") 
		
	    #### Supervised Initilisation vs Semi-supervised main loop
		start_train_time = time.time()  
		if epoch < supervised_init:
			print("Supervised Initilisation:", (epoch+1), "/" , supervised_init )
			for i in range(supervised_init):
				global_step = helpers.train_sup(train_loader, model, optimizer, epoch, global_step, args)                     
		if epoch >= supervised_init:
			global_step = helpers.train_semi(train_loader_l, train_loader_u, model, optimizer, epoch, global_step, args)  

		end_train_time = time.time()
		print("Evaluating the primary model:", end=" ")
		prec1, prec5 = helpers.validate(eval_loader, model, args, global_step, epoch + 1, num_classes = args.num_classes)

		# epoch_results[epoch,0] = epoch
		epoch_results[epoch,1] = prec1
		epoch_results[epoch,2] = prec5 
		# epoch_results[epoch,3] = dataset.acc
		# epoch_results[epoch,4] = time.time() - start_epoch_time
		# epoch_results[epoch,5] = end_train_time - start_train_time
		writer.add_scalar('prec1', prec1, epoch)
		writer.add_scalar('prec5', prec5, epoch)		

		# np.savetxt(args.file,epoch_results)
		
		#### Saving Model
		# torch.save(model.state_dict(), Path(args.logdir) / 'state_dict_epoch_{0:03d}.pth'.format(epoch))
		if (epoch % 10) == 0: 
		    torch.save(model.state_dict(), Path(args.logdir) / 'state_dict_epoch_{0:03d}.pth'.format(epoch))



if __name__ == '__main__':
    laplacenet()



