"""
2017-11-28T12:54:19.442Z
-----------------------------------------------
BLASSEL Luc
class selection module used in cifar10.py
-----------------------------------------------
"""
import pandas as pd
import numpy as np

root = 'data/CIFAR-10/cifar-10-batches-py/'
label1 = b'dog' #these must be binary -> don't forget b'name'
label2 = b'truck'
labels = [label1,label2]

def unpickle(file):
    """deserializes dataset"""
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def select(data,labels):
    """creates dataframe of images corresponding only to wanted labels"""
    label_names = unpickle(root+'batches.meta')
    wanted_labels = []
    for i in labels:
        wanted_labels.append(label_names[b'label_names'].index(i))
    # wanted_labels = [label_names[b'label_names'].index(label1),label_names[b'label_names'].index(label2)]

    df_data = pd.DataFrame(data[b'data'])
    df_labels = pd.DataFrame(data[b'labels'])
    df_labels.columns = ['labels']

    tot = pd.concat([df_labels,df_data],axis=1)

    sel = tot[tot['labels'].isin(wanted_labels)]

    dictionary = {}
    dictionary[b'labels'] = list(sel['labels'])
    dictionary[b'data'] = np.array(sel.drop('labels',axis=1))

    return dictionary,len(dictionary[b'labels'])

def main():
    sel = select(label1,label2,filename)

if __name__ == "__main__":
    main()
