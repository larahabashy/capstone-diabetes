# author: Peter Yang
# date: 2021-05-04

'''This script does train validation test split on image folders
for data loading in base Pytorch.

Usage: image_folder_split.py --positive_dir=<positive_dir> --negative_dir=<negative_dir> --train_size=<train_size> --valid_size=<valid_size> --seed=<seed>

Options:
--positive_dir=<positive_dir>   Relative path to the image folder of positive class(string)
--negative_dir=<negative_dir>   Relative path to the image folder of negative class(string)
--train_size=<train_size>       training data size(float)
--valid_size=<train_size>       validation data size(float)
--seed=<seed>                   random seed(integer)
'''

import numpy as np
import shutil
import os
from docopt import docopt
import traceback

opt = docopt(__doc__)

def main(positive_dir, negative_dir, train_size, valid_size, seed):
    
    #catch erroneous input
    try:
        int(seed)
    except Exception:
        traceback.print_exc()
        
    try:
        float(train_size)
    except Exception:
        traceback.print_exc()
        
    try:
        float(valid_size)
    except Exception:
        traceback.print_exc()
        
    if not float(train_size) + float(valid_size) < 1:
        raise ValueError('train size plus valid size is larger than or equal to 1.')
    
    #the directory to save images to
    root_dir = 'data'
    posCls = '/positive'
    negCls = '/negative'
    
    #remove data directory if exists, be careful if there already exists a 'data' directory
    if os.path.exists('data'):
        shutil.rmtree('data')


    os.makedirs(root_dir +'/train' + posCls)
    os.makedirs(root_dir +'/train' + negCls)
    os.makedirs(root_dir +'/val' + posCls)
    os.makedirs(root_dir +'/val' + negCls)
    os.makedirs(root_dir +'/test' + posCls)
    os.makedirs(root_dir +'/test' + negCls)


    # shuffle and split data
    np.random.seed(int(seed))
    for directory in [positive_dir, negative_dir]:

        allFileNames = os.listdir(directory)
        np.random.shuffle(allFileNames)
        train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                                          [int(len(allFileNames)*float(train_size)), int(len(allFileNames)*(float(train_size)+float(valid_size)))])


        train_FileNames = [directory+'/'+ name for name in train_FileNames.tolist()]
        val_FileNames = [directory+'/' + name for name in val_FileNames.tolist()]
        test_FileNames = [directory+'/' + name for name in test_FileNames.tolist()]
        
        if directory == positive_dir:
            print('positive class: ')
        else:
            print('negative class: ')
        print('Total images: ', len(allFileNames))
        print('Training: ', len(train_FileNames))
        print('Validation: ', len(val_FileNames))
        print('Testing: ', len(test_FileNames))
        print()

        # Copy-pasting images
        if directory == positive_dir:
            for name in train_FileNames:
                shutil.copy(name, root_dir +'/train' + posCls)
                
            for name in val_FileNames:
                shutil.copy(name, root_dir +'/val' + posCls)

            for name in test_FileNames:
                shutil.copy(name, root_dir +'/test' + posCls)
                
        else:
            for name in train_FileNames:
                shutil.copy(name, root_dir +'/train' + negCls)
                
            for name in val_FileNames:
                shutil.copy(name, root_dir +'/val' + negCls)

            for name in test_FileNames:
                shutil.copy(name, root_dir +'/test' + negCls)

if __name__ == "__main__":
    main(opt["--positive_dir"], opt["--negative_dir"], opt["--train_size"], opt["--valid_size"], opt["--seed"])