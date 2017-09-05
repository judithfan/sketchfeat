
######### dependencies #######
import numpy as np
import utils
import pickle
import os
import re

import tensorflow as tf
import vgg16

######### constants #########
# required vgg image sizes 
VGG_SIZE_X = 224
VGG_SIZE_Y = 224
VGG_SIZE_Z = 3


SKETCH_FOLDER = './sketch_data'


# parses a foldername to return the relevant info. returns -1 if file is not a png or the filename is ill-formed
def parseFileName(filename):
    
    # ignore DS_Store files 
    if filename == ".DS_Store": 
        return -1
    
    
    fileInfo = filename.split("_")
    
    [target, fileType] = fileInfo[-1].split(".")
    
    # ignore csv's 
    if fileType == "csv": 
        return -1 
    
    if fileType == "png":
        
        subjectID = fileInfo[0] 
        trial = fileInfo[-2]
        
        
        return (subjectID, trial, target) 
        
    print "error: filename syntax incorrect" 
    return -1
        

# takes a batch and makes them into manageable 160 rows 
def splitBatches(full_batch, sz):
    num_batches = full_batch.shape[0]/sz; 
    num_remainder = full_batch.shape[0]%sz

    batch = []

    for batch_i in xrange(0,num_batches):
        batch.append(full_batch[batch_i * sz: (batch_i + 1) * sz])


    if num_remainder != 0: 
        batch.append(full_batch[(-1 * num_remainder):])
        
        
    return batch


full_batch = np.empty((0, VGG_SIZE_X, VGG_SIZE_Y, VGG_SIZE_Z), float)
target = [] 
for folderName, subfolders, filenames in os.walk(SKETCH_FOLDER):
    print ('Downloading sketches from: '  + folderName)
    
    # skip the sketch_data folder. 
    if folderName == SKETCH_FOLDER: 
        continue
    
    

    for filename in filenames: 
        
        
        if parseFileName(filename) != -1: 
            [subjectID_i, trial_i, target_i] = parseFileName(filename)
            target.append(target_i)
            
            img = utils.load_image(folderName + '/' + filename)
            
            # take out the fourth dimension, alpha, which controls transparency
            img = img[:,:,:3]
            
            img = img.reshape(1, VGG_SIZE_X, VGG_SIZE_Y, VGG_SIZE_Z)
            full_batch = np.concatenate((full_batch, img))            
        
#         print ('FILE INSIDE ' + folderName + ':' + filename) 



# running the gpu 

for tf_run in xrange(0, len(batch)):
    batch_mini = batch[tf_run]

    print "running batch" + str(tf_run)
    print ".........." 
          
    with tf.device('/gpu:0'):
        with tf.Session() as sess:
            image = tf.placeholder("float", [batch_mini.shape[0], VGG_SIZE_X, VGG_SIZE_Y, VGG_SIZE_Z])

            feed_dict = {image: batch_mini}

            vgg = vgg16.Vgg16()
            with tf.name_scope("content_vgg"):
                vgg.build(image)

            act_wanted = [vgg.pool1, vgg.pool2, vgg.pool3, vgg.pool4, vgg.pool5, vgg.fc6, vgg.fc7] # , vgg.prob]
            act = sess.run(act_wanted, feed_dict=feed_dict)


    print "completed batch" 
    print "now saving...."
    pickle.dump(act, open( "/tigress/rslee/sketch_act" + str(tf_run) + ".p", "wb" ) )
    print "saved vgg for batch" + str(tf_run)
