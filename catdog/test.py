"""
test.py
~~~~~~~

test from weight checkpoint and write result to csv file
"""

# Import libraries
import csv

from data import fetch_test_data
from models import conv_simple

# constants
out_file = 'result.csv'


# test function
def run_test():
    # fetch data
    test = fetch_test_data()

    # bring model
    model = conv_simple()

    # run test
    predicts = model.predict(test, verbose=1)

    # write to output file
    with open(out_file, 'w') as csv_file:
        csv_file_obj = csv.writer(csv_file, dialect='excel')
        csv_file_obj.writerow(['id','label'])

        for i in range(len(predicts)):
            predict = predicts[i][0]
            csv_file_obj.writerow([i+1, predict])


# actually run test
run_test()