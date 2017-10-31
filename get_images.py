# pulls images from the amazon website
######### dependencies #######
import numpy as np
import utils
import pickle
import urllib



######### constants #########
# required vgg image sizes 
VGG_SIZE_X = 224
VGG_SIZE_Y = 224
VGG_SIZE_Z = 3

# constants for the images
NUM_VIEWS = 40


# pull data from online 
cars = ['limoToSUV_10','limoToSUV_99','smartToSedan_10','smartToSedan_99'];
furnitures = ['bedChair_1', 'bedChair_100', 'tableBench_1', 'tableBench_100'];

# sample webpage: https://s3.amazonaws.com/morphrecog-images-1/bedChair_1_20.png.png

for obj in cars + furnitures:
    for view in xrange(0,NUM_VIEWS):
        imgname = obj + '_' + str(view) + '.png.png'
        imgloc ='https://s3.amazonaws.com/morphrecog-images-1/' + imgname
        urllib.urlretrieve(imgloc, './object_data/' + imgname)



