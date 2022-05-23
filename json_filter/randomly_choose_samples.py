# randomly choose samples and create 2 new json files by splitting the samples

import numpy as np
import json
import random
import os

sample_size = 1024
set_index = 7

# prepare sampling index
dataset_size = 37880 - sample_size * (set_index - 1)   
dataset_indices = list(range(dataset_size))
np.random.shuffle(dataset_indices)
dataset_indices = dataset_indices[0:int(sample_size)]

# with open('/../../media/16TBHDD/leong/dataset/star/new/explore_datasets/thestar_304/rendered/cr_train/TRAIN/cr_dataset_meta.json') as fp:
#     data = json.load(fp)
with open("active_learning_labeling/json_samples/remain_from_" + str(sample_size) + "_set_" + str(set_index-1) + ".json") as fp:
    data = json.load(fp)

counter = 0
t_counter = 0
v_counter = 0

sampled_set = "active_learning_labeling/json_samples/ori_sample_" + str(sample_size) + "_set_" + str(set_index) + ".json"
remain_set = "active_learning_labeling/json_samples/remain_from_" + str(sample_size) + "_set_" + str(set_index) + ".json"

print("working ... ")
with open(sampled_set, 'a+') as sample, open(remain_set, 'a+') as remain:
    for item in data['images']:          
        if item.get('chipspec'):
            for row in item['chipspec']:
                if (item['chipspec'][row] == 'ROUND'):                    
                    if counter in dataset_indices:
                        json_formatted_str = json.dumps(item, indent=4)                        
                        # print("{},".format(json_formatted_str))
                        if (t_counter + 1) != sample_size:
                            sample.write("{},\n".format(json_formatted_str))   
                        else:
                            sample.write("{}".format(json_formatted_str))
                        t_counter += 1                     
                    else:
                        json_formatted_str = json.dumps(item, indent=4)
                        # print("{},".format(json_formatted_str))
                        if (v_counter + 1) != (dataset_size - sample_size):
                            remain.write("{},\n".format(json_formatted_str)) 
                        else:
                            remain.write("{}".format(json_formatted_str)) 
                        v_counter += 1                      
                    counter += 1
    remain.close()
    sample.close()

print("Number of all samples: ", counter)
print("sampled: ", t_counter)
print("remain: ", v_counter)

# merge files
file_list_to_merge = [sampled_set, remain_set]

for i in range(len(file_list_to_merge)):
    filenames = ['toy_json/to_merge1.json', file_list_to_merge[i]]
    
    # Open file3 in write mode
    with open('active_learning_labeling/json_samples/inter.json', 'w') as outfile:
    
        # Iterate through list
        for names in filenames:
    
            # Open each file in read mode
            with open(names) as infile:
    
                # read the data from file1 and
                # file2 and write it in file3
                outfile.write(infile.read())
    
            # Add '\n' to enter data of file2
            # from next line
            outfile.write("\n")


    # repeat for the lst file
    filenames = ['active_learning_labeling/json_samples/inter.json', 'toy_json/to_merge2.json']
    
    # Open file3 in write mode
    with open(file_list_to_merge[i], 'w') as outfile:
    
        # Iterate through list
        for names in filenames:
    
            # Open each file in read mode
            with open(names) as infile:
    
                # read the data from file1 and
                # file2 and write it in file3
                outfile.write(infile.read())

