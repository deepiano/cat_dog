import os
from os import listdir
from os.path import join, isfile

import re

import cv2
import numpy as np


file_dir_path_cat = '/home/bong/MyProject/cat_dog/dataAll/cat'
file_dir_path_dog = '/home/bong/MyProject/cat_dog/dataAll/dog'
test_dir_path = '/home/bong/MyProject/cat_dog/test'

train_dir_path = '/home/bong/MyProject/cat_dog/data/train/cat'
val_dir_path = '/home/bong/MyProject/cat_dog/data/val/cat'



def main():
   
#    get_max_min_width_height()
   calculate_mean_and_std()


    
def tryint(s):
    try:
        return int(s)
    except:
        return s


def alphanum_key(s):
    return [tryint(c) for c in re.split('([0-9]+)', s)]


def sort_nicely(l):
    l.sort(key=alphanum_key)


def get_file_name_list_nicely_sorted(file_dir_path):

    list_file_name = [f for f in listdir(file_dir_path)\
                        if isfile(join(file_dir_path, f))\
                        and f.endswith('.jpg')]
    
    sort_nicely(list_file_name)
    
    return list_file_name


def get_max_min_width_height():


    print('=> exploring cat') 
    list_file_name_cat = [f for f in listdir(file_dir_path_cat)\
                        if isfile(join(file_dir_path_cat, f))\
                        and f.endswith('.jpg')]
    
    sort_nicely(list_file_name_cat)
    
    min_width = 9999
    max_width = 0
    
    min_height = 9999
    max_height = 0

    min_height_name = ""
    min_width_name = ""

    num_small_img = 0

    sum_width = 0
    sum_height = 0
    
    num_img = 0
    for name in list_file_name_cat:
    
        cur_file_path = os.path.join(file_dir_path_cat, name)
        img = cv2.imread(cur_file_path)
        height = img.shape[0]
        width = img.shape[1]

#        sum_width += width
#        sum_height += height 
#        num_img += 1

        if width < 64 or height < 64 :
            num_small_img += 1
        
#        if height < min_height :
#            min_height_name = name
#        if width < min_width :
#            min_width_name = name
#
#        min_height = min(min_height, height)
#        max_height = max(max_height, height)
#    
#        min_width = min(min_width ,width)
#        max_width = max(max_width, width)

    print('=> exploring dog') 
    list_file_name_dog = [f for f in listdir(file_dir_path_dog)\
                        if isfile(join(file_dir_path_dog, f))\
                        and f.endswith('.jpg')]
    
    sort_nicely(list_file_name_dog)
    
    for name in list_file_name_dog:
    
        cur_file_path = os.path.join(file_dir_path_dog, name)
        img = cv2.imread(cur_file_path)
        height = img.shape[0]
        width = img.shape[1]

#        sum_width += width
#        sum_height += height 
#        num_img += 1

        if width < 64 or height < 64 :
            num_small_img += 1
    
#    print ('width mean : ', (float(sum_width) / num_img))
#    print ('height mean : ', (float(sum_height) / num_img))

    print('num : ', num_small_img)

#        if height < min_height :
#            min_height_name = name
#        if width < min_width :
#            min_width_name = name
#
#        min_height = min(min_height, height)
#        max_height = max(max_height, height)
#    
#        min_width = min(min_width ,width)
#        max_width = max(max_width, width)
#
#    print('min_width : ', min_width)
#    print('min_height : ', min_height)
#
#    print('max_width : ', max_width)
#    print('max_height : ', max_width)
#
#    print('min_width_name : ', min_width_name)
#    print('min_height_name : ', min_height_name)


def calculate_mean_and_std():

    IMG_SIZE = 256 

    list_file_name_cat = get_file_name_list_nicely_sorted(file_dir_path_cat) 
    list_file_name_dog = get_file_name_list_nicely_sorted(file_dir_path_dog) 
    
    num_img = 0
    sum_img = np.zeros([IMG_SIZE, IMG_SIZE, 3])
    
    # Calculate mean
    for name in list_file_name_cat:
    
        num_img += 1
        cur_file_path = os.path.join(file_dir_path_cat, name)
        img = cv2.imread(cur_file_path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        sum_img += img

    for name in list_file_name_dog:
    
        num_img += 1
        cur_file_path = os.path.join(file_dir_path_dog, name)
        img = cv2.imread(cur_file_path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        sum_img += img

    data_mean = np.mean(sum_img / num_img, axis=(0, 1))

    sum_img = np.zeros([IMG_SIZE, IMG_SIZE, 3])

    # Calculate standard deviation
    for name in list_file_name_cat:
    
        cur_file_path = os.path.join(file_dir_path_cat, name)
        img = cv2.imread(cur_file_path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        deviation_square = np.square(img - data_mean)
        sum_img += deviation_square 

    for name in list_file_name_dog:
    
        cur_file_path = os.path.join(file_dir_path_dog, name)
        img = cv2.imread(cur_file_path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        deviation_square = np.square(img - data_mean)
        sum_img += deviation_square 


    data_std = np.mean(sum_img / num_img, axis=(0, 1)) 
    data_std = np.sqrt(data_std)

    print('mean : ', data_mean)
    print('std  : ', data_std)

     

if __name__=='__main__':
    main()
