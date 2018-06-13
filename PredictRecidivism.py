import sys
import json
import operator

import matplotlib.pyplot as plt
from keras.models import load_model
import numpy as np
from scipy.stats import linregress


COUNTY_INDEX = 1
GENDER_INDEX = 2
STATUS_INDEX = -1

counties_dict = json.load(open('counties.dict'))

def abline(slope, intercept):
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')


def denormalize(d):
    d *= [4., 62.,  1., 84.,  1.]
    d += [2008., 0., 0., 16., 0.]

    return d


def analyze(predicted, demographics, stat):
    X = []
    Y = []
    names = []

    for value in predicted:
        if value[0] == 'UNKNOWN':
            continue
        print(value[0])
        names.append(value[0])
        X.append(value[1])

    for value in demographics:
        values = value.split()
        print(values)
        y = values[stat]
        y = float(y)
        Y.append(y)

    print(len(X), len(Y))


    regression = linregress(X, Y)
    fig, ax = plt.subplots()
    ax.plot(X, Y, 'ro')

    abline(regression[0], regression[1])
    print('r-squared: ', regression[2]**2)

    for i, txt in enumerate(names):
        ax.annotate(txt, (X[i], Y[i]))
    plt.xlabel('% Predicted Recidivism', fontsize=18)
    if stat == 1:
        plt.ylabel('Black Population %', fontsize=16)
    elif stat == 2:
        plt.ylabel('% Under Poverty Level', fontsize=16)
    elif stat == 3:
        plt.ylabel('Crime Rate / 100,000', fontsize=16)
    plt.show()


def get_predicted(model, data):
    val_split = int(data.shape[0] * .75)
    predictions = model.predict(data[val_split:, :STATUS_INDEX])

    denormalized_data = denormalize(data[val_split:])
    predictions = np.where(predictions > 0.5, 1, 0)

    predicted_data = []

    for i, prediction in enumerate(predictions):
        info = list(np.append(denormalized_data[i], prediction))
        predicted_data.append(info)

    county_predicted = dict()
    county_totals = dict()

    for datum in predicted_data:
        county = datum[COUNTY_INDEX]
        if datum[-1] == 1:
            if county not in county_predicted:
                county_predicted[county] = 1
            else:
                county_predicted[county] += 1
        if county not in county_totals:
            county_totals[county] = 1
        else:
            county_totals[county] += 1

    percent_predicted = dict()
    for key, value in county_predicted.items():
        total = county_totals[key]
        county_name = [k for k, v in counties_dict.items() if v == key][0]
        percent_predicted[county_name] = value / total

    total_correct = 0
    for datum in predicted_data:
        if 0 == datum[-2]:
            total_correct += 1

    print('NN acc: ', total_correct / len(predicted_data))
    get_stats(predicted_data)

    return sorted(percent_predicted.items(), key=lambda x:x[0].lower())

def algorithm_predict(data):
    stats = open('RecidivismStats.dat').readlines()

    denormalized_data = denormalize(data)

    male_percent = float(stats[0].split()[1])
    female_percent = float(stats[1].split()[1])

    stats_dict = dict()

    predicted_data = []

    for line in stats[2:]:
        county_name = line.split()[0].upper()
        stats_dict[county_name] = float(line.split()[1])

    for i, datum in enumerate(data):
        county_name = [k for k, v in counties_dict.items() if v == datum[COUNTY_INDEX]][0].replace(' ', '')
        if county_name == 'UNKNOWN':
            continue

        gender_percent = male_percent if datum[GENDER_INDEX] == 0 else female_percent

        if stats_dict[county_name] + gender_percent  > .91:
            info = list(np.append(denormalized_data[i], 1))
            predicted_data.append(info)
        else:
            info = list(np.append(denormalized_data[i], 0))
            predicted_data.append(info)

    total_correct = 0
    for datum in predicted_data:
        if datum[-1] == datum[-2]:
            total_correct += 1

    print(total_correct / len(predicted_data))

# Used to make RecidivismStats.dat
def get_stats(data):
    stats_totals = dict()
    stats = dict()

    gender_total = {'MALE': 0, 'FEMALE': 0}
    gender_stats = {'MALE': 0, 'FEMALE': 0}
    gender_predicted = {'MALE': 0, 'FEMALE': 0}

    for datum in data:
        county_name = [k for k, v in counties_dict.items() if v == datum[COUNTY_INDEX]][0]

        if county_name in stats_totals:
            stats_totals[county_name] += 1
        else:
            stats_totals[county_name] = 1
        if datum[-2] == 1:
            if county_name not in stats:
                stats[county_name] = 1
            else:
                stats[county_name] += 1

        if datum[GENDER_INDEX] == 0:
            gender_total['MALE'] += 1
            if datum[-2] == 1:
                gender_stats['MALE'] += 1
            if datum[STATUS_INDEX] == 1:
                gender_predicted['MALE'] += 1
        else:
            gender_total['FEMALE'] += 1
            if datum[-2] == 1:
                gender_stats['FEMALE'] += 1
            if datum[STATUS_INDEX] == 1:
                gender_predicted['FEMALE'] += 1

    print('Male:', gender_stats['MALE'] / gender_total['MALE'])
    print('Female:', gender_stats['FEMALE'] / gender_total['FEMALE'])

    print('Male Predicted:', gender_predicted['MALE'] / gender_total['MALE'])
    print('Female Predicted:', gender_predicted['FEMALE'] / gender_total['FEMALE'])

    print('Male Total #:', gender_stats['MALE'], gender_predicted['MALE'], gender_total['MALE'])
    print('Female Total #:', gender_stats['FEMALE'], gender_predicted['FEMALE'], gender_total['FEMALE'])

    for key, value in stats_totals.items():
        total = stats_totals[key]
        if key in stats:
            print(key, stats[key] / total)
        

def main(command, network=True):

    model = load_model('recidivism_predictor.h5')
    data = np.load('Recidivism__Beginning_2008_in.npz')
    demographics = open('DemographicsNewYork.dat').readlines()[1:]

    if network == True:
        predictions = get_predicted(model, data)
    else:
        predictions = algorithm_predict(data)

    if command == 'race':
        analyze(predictions, demographics, 1)
    if command == 'poverty':
        analyze(predictions, demographics, 2)
    if command == 'crime':
        analyze(predictions, demographics, 3)


if len(sys.argv) > 2:
    main(sys.argv[1], sys.argv[2])
else:
    main(sys.argv[1])




