import sys
import json

import numpy as np
from keras import models, layers

COUNTY_INDEX = 1
GENDER_INDEX = 2
STATUS_INDEX = -1

csv_file = sys.argv[1]
num_epochs = 20


def normalize(d):
    d -= np.min(d, axis=0)
    d /= np.ptp(d, axis=0)
    return d


def to_int_dict(data, index):
    int_dict = dict()
    for datum in data:
        key = datum[index]
        if key not in int_dict:
            int_dict[key] = len(int_dict)
    return int_dict


def write_data_to_file(data):
    with open('{}_in.npz'.format(csv_file), 'wb') as file:
        np.save(file, data)
            


def create_input_arrays(csv_file):
    data = np.genfromtxt('{}.csv'.format(csv_file), delimiter=',', dtype=None, encoding=None)

    data = data[1:]
    counties = to_int_dict(data, COUNTY_INDEX)
    genders = to_int_dict(data, GENDER_INDEX)
    status = to_int_dict(data, STATUS_INDEX)
    json.dump(counties, open('counties.dict', 'w'))

    for i, datum in enumerate(data):
        data[i][COUNTY_INDEX] = counties[datum[COUNTY_INDEX]]
        data[i][GENDER_INDEX] = genders[datum[GENDER_INDEX]]
        status_num = status[datum[STATUS_INDEX]]
        data[i][STATUS_INDEX] = 0 if status_num == 0 else 1

    data = np.ndarray.astype(data, dtype=np.float32)
    data = normalize(data)
    np.random.shuffle(data)

    write_data_to_file(data)

    return data


input_arrays = create_input_arrays(csv_file)

# input_data, targets = input_arrays[:, :STATUS_INDEX], input_arrays[:, STATUS_INDEX]

# model = models.Sequential()
# model.add(layers.Dense(128, activation='relu',
#                        input_shape=(input_data.shape[1],)))
# model.add(layers.Dense(128, activation='relu'))
# model.add(layers.Dense(16, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))
# print(model.summary())

# model.compile(optimizer='adam', loss='binary_crossentropy',
#               metrics=['accuracy'])

# history = model.fit(
#     input_data,
#     targets,
#     validation_split=.25,
#     epochs=num_epochs,
#     batch_size=128)

# model.save('recidivism_predictor_improved.h5')




