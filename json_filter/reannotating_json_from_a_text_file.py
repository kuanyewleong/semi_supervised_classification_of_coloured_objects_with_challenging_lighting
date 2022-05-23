import json


sample_size = str(1024)
set_index = str(7)

with open('/home/leong/color-recognition/active_learning_labeling/json_samples/ori_sample_' + sample_size + "_set_" + set_index + '.json') as fp:
    data = json.load(fp)

counter = 0
new_set = '/home/leong/color-recognition/active_learning_labeling/json_samples/sample_' + sample_size + '_predicted_label_set_' + set_index + '.json'

# new_class = [0, 12, 7, 6]
new_class = []
with open('/home/leong/color-recognition/active_learning_labeling/predicted_labels/predicted_labels_' + sample_size + "_set_" + set_index + '.txt') as f:
    for line in f:
        new_class.append(int(line))

new_color = ['PINK', 'BLUE', 'VIOLET',\
        'RED', 'LIGHT_BLUE', 'GREEN', \
        'BLACK','ORANGE', 'YELLOW', \
        'DARK_BLUE', 'RED', 'GREEN', \
        'BLACK', 'ORANGE', 'YELLOW']
    
new_default_nn = ['DEFAULT', 'DEFAULT', 'DEFAULT',\
        'DEFAULT', 'DEFAULT', 'DEFAULT', \
        'DEFAULT','DEFAULT', 'DEFAULT', \
        'DEFAULT', 'NN', 'NN', \
        'NN', 'NN', 'NN']

item_amount = 0
for item in data['images']: 
    item_amount += 1

item_counter = 0
with open(new_set, 'a+') as new:
    for item in data['images']:  
        item_counter += 1  
        if item.get('chipspec'):
            for row in item['chipspec']:                                                   
                item['chipspec']["color"] = new_color[new_class[counter]]
                item['chipspec']["design"] = new_default_nn[new_class[counter]]
                json_formatted_str = json.dumps(item, indent=4)
                # print(json_formatted_str)
                # print("{},".format(json_formatted_str))
                if item_amount != item_counter:
                    new.write("{},\n".format(json_formatted_str))
                else:
                    new.write("{}".format(json_formatted_str))
                counter += 1    
                break
    new.close()

print("Number of all samples: ", counter)

