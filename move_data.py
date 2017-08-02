import os
import shutil
from os import listdir
from os.path import isfile, join

import re

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    return [tryint(c) for c in re.split('([0-9]+)', s)]

def sort_nicely(l):
    l.sort(key=alphanum_key)


file_dir_path = '/home/bong/MyProject/cat_dog/dataAll/cat'
train_dir_path = '/home/bong/MyProject/cat_dog/data/train/cat'
val_dir_path = '/home/bong/MyProject/cat_dog/data/val/cat'
list_file_name = [f for f in listdir(file_dir_path)\
                    if isfile(join(file_dir_path, f))\
                    and f.endswith('.jpg')]

sort_nicely(list_file_name)
for i, f in enumerate(list_file_name):
    file_path = join(file_dir_path, f)
    name = f.replace('.', '_', 1)
    if i < 10000:  
        save_path = join(train_dir_path, name)
        shutil.copy(file_path, save_path) 
    else:
        save_path = join(val_dir_path, name)
        shutil.copy(file_path, save_path) 


"""
file_dir_path = '/home/bong/MyProject/cat_dog/data/dog'
train_dir_path = '/home/bong/MyProject/cat_dog/data/train/dog'
val_dir_path = '/home/bong/MyProject/cat_dog/data/val/dog'
list_file_name = [f for f in listdir(file_dir_path)\
                    if isfile(join(file_dir_path, f))\
                    and f.endswith('.jpg')]

sort_nicely(list_file_name)
#print(list_file_name)
for i, f in enumerate(list_file_name):
    file_path = join(file_dir_path, f)
    name = f.replace('.', '_', 1)
    if i < 10000:  
        save_path = join(train_dir_path, name)
        shutil.copy(file_path, save_path) 
    else:
        save_path = join(val_dir_path, name)
        shutil.copy(file_path, save_path) 
"""
