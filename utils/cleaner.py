# This is a sample Python script.
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import csv
import os


# Temporal implementation of cleaning
def clean(df):
    filtered_colns = ['Index', 'Address', ' ERC20 most sent token type', ' ERC20_most_rec_token_type']
    df = df.drop(filtered_colns, axis=1)
    df.fillna(df.median(), inplace=True)
    return df


def fill_missing():
    csv_file2 = '/users/saksham/Desktop/transaction_dataset23.csv'
    with open(csv_file2) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count== 0:
                print(f'Column names are {", ".join(row)}')
                j = 0
                while(j < 51):
                    print(str(row[j]) + " " + str(j))
                    j = j + 1
            # elif:
            else:
                print(line_count)
                i = 0
                while(i < 51):
                    if row[i] != '':
                        print(str(row[i]) + "," + str(i))
                    else:
                        if not (os.system('wget https://https://etherscan.io/address/' + str(row[0]) + '#tokentxns')):
                            df = pd.read_csv(csv_file2, delimiter=",")
                            df.loc[line_count:48] = '0.0'
                            df.to_csv(csv_file2)
                            #row[i] == '0.0'
                            #print(row[i])
                        if i < 49:
                            #df.to_csv(csv_file2)
                            print(row[i])
                        else:
                            df = pd.read_csv(csv_file2, delimiter=",")
                            print(row[i])
                            print(df[line_count, [i]])
                            #df.loc[line_count, 'ERC20 most sent token type'] = 'None'
                            #df.to_csv(csv_file2)
                    i = i + 1

            line_count += 1
            print(f'Processed {line_count} lines.')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    fill_missing()