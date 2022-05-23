import json


with open('/home/leong/color-recognition/toy_json/toy_data_round.json') as fp:
    data = json.load(fp)

counter = 0
new_set = '/home/leong/color-recognition/toy_json/edited_toy_data.json'

new_class = [0, 12, 7, 6]
new_color = ['ROUND-DEFAULT-PINK', 'ROUND-DEFAULT-BLUE', 'ROUND-DEFAULT-VIOLET',\
        'ROUND-DEFAULT-RED', 'ROUND-DEFAULT-LIGHT_BLUE', 'ROUND-DEFAULT-GREEN', \
        'ROUND-DEFAULT-BLACK',' ROUND-DEFAULT-ORANGE', 'ROUND-DEFAULT-YELLOW', \
        'ROUND-DEFAULT-DARK_BLUE', 'ROUND-NN-RED', 'ROUND-NN-GREEN', \
        'ROUND-NN-BLACK', 'ROUND-NN-ORANGE', 'ROUND-NN-YELLOW']

with open(new_set, 'a+') as new:
    for item in data['images']:    
        if item.get('chipspec'):
            for row in item['chipspec']:                                                   
                item['chipspec']["color"] = new_color[new_class[counter]]
                json_formatted_str = json.dumps(item, indent=4)
                # print(json_formatted_str)
                print("{},".format(json_formatted_str))
                new.write("{},\n".format(json_formatted_str))
                counter += 1    
                break
    new.close()

print("Number of all samples: ", counter)

