import os

# sample_size = 1024
# set_index = 7

# source_file = 'active_learning_labeling/json_samples/sample_' + str(sample_size) + '_predicted_label_set_' + str(set_index) + '.json'
# Creating a list of filenames
source_file = "area3_trainset.json"
filenames = ['toy_json/to_merge1.json', source_file]

# Open file3 in write mode
with open('toy_json/inter.json', 'w') as outfile:

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

to_do = "rm " + source_file
os.system(to_do)

# repeat for the lst file
filenames = ['toy_json/inter.json', 'toy_json/to_merge2.json']

# Open file3 in write mode
with open(source_file, 'w') as outfile:

    # Iterate through list
    for names in filenames:

        # Open each file in read mode
        with open(names) as infile:

            # read the data from file1 and
            # file2 and write it in file3
            outfile.write(infile.read())

        # Add '\n' to enter data of file2
        # from next line
        # outfile.write("\n")

to_do = "cp " + source_file + " /../../media/16TBHDD/leong/dataset/star/new/from_han/CR_curved_train_3/"
os.system(to_do)


