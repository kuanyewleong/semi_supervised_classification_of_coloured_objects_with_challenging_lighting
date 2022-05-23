# make folders for all classes
import os

target_names = ['ROUND-DEFAULT-PINK', 'ROUND-DEFAULT-BLUE', 'ROUND-DEFAULT-VIOLET',\
        'ROUND-DEFAULT-RED', 'ROUND-DEFAULT-LIGHT_BLUE', 'ROUND-DEFAULT-GREEN', \
        'ROUND-DEFAULT-BLACK',' ROUND-DEFAULT-ORANGE', 'ROUND-DEFAULT-YELLOW', \
        'ROUND-DEFAULT-DARK_BLUE', 'ROUND-NN-RED', 'ROUND-NN-GREEN', \
        'ROUND-NN-BLACK', 'ROUND-NN-ORANGE', 'ROUND-NN-YELLOW']

for i in range(len(target_names)):
    command = "mkdir " + target_names[i]
    os.system(command)

# --------------------------------------------------------------------------------




# # make balanced class labels (single folder)
# import glob
# import numpy as np

# dataset_size = 81607 # remember to change according to the sample size
# indices = list(range(dataset_size))
# np.random.shuffle(indices)        
# indices = indices[0:300] # [0:x], for x = desired_percentage * total_train_size / 15
# indices.sort() 

# # get the path/directory
# classLabel = 'ROUND-NN-YELLOW'
# folder_dir = 'train+val/' + classLabel
 
# # iterate over files in
# # that directory
# counter = 0
# for images in glob.iglob(f'{folder_dir}/*'):
   
#     # check if the image seequence match the random indice prepared
#     if counter in indices:
#         fileName = images.replace(f'{folder_dir}/', '')
#         print(fileName)
#         with open("4500_balanced_labels/01.txt","a+") as file:
#             fileName_classLabel = fileName + " " + classLabel
#             file.write(fileName_classLabel)
#             file.write("\n")
#     counter += 1    

# -----------------------------------------------------------------------------------


# make balanced class labels (multiple folders)
import glob
import numpy as np

classLabel = ['ROUND-DEFAULT-PINK', 'ROUND-DEFAULT-BLUE', 'ROUND-DEFAULT-VIOLET',\
        'ROUND-DEFAULT-RED', 'ROUND-DEFAULT-LIGHT_BLUE', 'ROUND-DEFAULT-GREEN', \
        'ROUND-DEFAULT-BLACK','ROUND-DEFAULT-ORANGE', 'ROUND-DEFAULT-YELLOW', \
        'ROUND-DEFAULT-DARK_BLUE', 'ROUND-NN-RED', 'ROUND-NN-GREEN', \
        'ROUND-NN-BLACK', 'ROUND-NN-ORANGE', 'ROUND-NN-YELLOW']

# first get the size of images in each folder
classSize = []
for i in range(len(classLabel)):
    img_counter = 0
    folder_dir = 'train/' + classLabel[i]
    for images in glob.iglob(f'{folder_dir}/*'):
        img_counter += 1
    classSize.append(img_counter)
    # print(img_counter)    
    
for i in range(len(classLabel)):
    dataset_size = classSize[i] # now apply the sample size
    indices = list(range(dataset_size))
    np.random.shuffle(indices)        
    indices = indices[0:4432] # [0:x], for x = desired_percentage * total_train_size / 15
    indices.sort() 

    counter = 0
    folder_dir = 'train/' + classLabel[i]
    for images in glob.iglob(f'{folder_dir}/*'):   
        # check if the image seequence match the random indice prepared           
        if counter in indices:
            fileName = images.replace(f'{folder_dir}/', '')
            print(fileName)
            with open("labels/66480_balanced_labels_20perccent/02.txt","a+") as txtfile:
                fileName_classLabel = fileName + " " + classLabel[i]
                txtfile.write(fileName_classLabel)
                txtfile.write("\n")
            txtfile.close()  
        counter += 1 
        

print(sum(classSize))