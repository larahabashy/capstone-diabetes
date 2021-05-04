# author: Peter Yang
# date: 2021-05-04

'''This script does train test split on image folders for
data loading in base Pytorch later.

Usage: image_folder_split.py --positive_dir=<positive_dir> --negative_dir=<negative_dir> --train_size=<train_size> --seed=<seed>

Options:
--positive_dir=<positive_dir>   Path to the image folder of positive class(string)
--negative_dir=<negative_dir>   Path to the image folder of negative class(string)
--train_size=<train_size>       training data size(inetger)
--seed=<seed>                   random seed(integer)
'''

import numpy as np
import shutil
import os
from docopt import docopt

opt = docopt(__doc__)

def main(positive_dir, negative_dir, train_size, seed):
    
    #the directory to save images to
    root_dir = 'data'
    posCls = '/positive'
    negCls = '/negative'
    
    #remove data directory if exists, be careful if there already exists a 'data' directory
    if os.path.exists('data'):
        shutil.rmtree('data')


    os.makedirs(root_dir +'/train' + posCls)
    os.makedirs(root_dir +'/train' + negCls)
    os.makedirs(root_dir +'/test' + posCls)
    os.makedirs(root_dir +'/test' + negCls)


    # shuffle and split data
    np.random.seed(int(seed))
    for directory in [positive_dir, negative_dir]:

        allFileNames = os.listdir(directory)
        np.random.shuffle(allFileNames)
        train_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                                          [int(len(allFileNames)*float(train_size))])


        train_FileNames = [directory+'/'+ name for name in train_FileNames.tolist()]
        test_FileNames = [directory+'/' + name for name in test_FileNames.tolist()]
        
        if directory == positive_dir:
            print('positive class: ')
        else:
            print('negative class: ')
        print('Total images: ', len(allFileNames))
        print('Training: ', len(train_FileNames))
        print('Testing: ', len(test_FileNames))
        print()

        # Copy-pasting images
        if directory == positive_dir:
            for name in train_FileNames:
                shutil.copy(name, root_dir +'/train' + posCls)

            for name in test_FileNames:
                shutil.copy(name, root_dir +'/test' + posCls)
                
        else:
            for name in train_FileNames:
                shutil.copy(name, root_dir +'/train' + negCls)

            for name in test_FileNames:
                shutil.copy(name, root_dir +'/test' + negCls)

if __name__ == "__main__":
    main(opt["--positive_dir"], opt["--negative_dir"], opt["--train_size"], opt["--seed"])