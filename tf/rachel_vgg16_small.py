
######### dependencies #######
import numpy as np
import utils
import pickle

######### dependencies #######
import numpy as np
import utils

import tensorflow as tf
import vgg16

######### constants #########
# required vgg image sizes 
VGG_SIZE_X = 224
VGG_SIZE_Y = 224
VGG_SIZE_Z = 3

# constants for the images
NUM_VIEWS = 40 


# to upload multiple images

cars = ['limoToSUV_10','limoToSUV_99','smartToSedan_10','smartToSedan_99'];
furnitures = ['bedChair_1', 'bedChair_100', 'tableBench_1', 'tableBench_100']

batch = np.empty((0, VGG_SIZE_X, VGG_SIZE_Y, VGG_SIZE_Z), float)
for obj in cars + furnitures:
    for view in xrange(0,NUM_VIEWS):
        imgname = obj + '_' + str(view) + '.png.png'
        imgloc = 'object_data/' + imgname
        img = utils.load_image(imgloc)
        img = img.reshape(1, VGG_SIZE_X, VGG_SIZE_Y, VGG_SIZE_Z)
        batch = np.concatenate((batch, img))
 # smaller batch for testing first
print batch.shape[0]
batch_mini = batch[:160,:,:,:]
print batch_mini.shape[0]




with tf.device('/gpu:0'):
    with tf.Session() as sess:
        image = tf.placeholder("float", [batch_mini.shape[0], VGG_SIZE_X, VGG_SIZE_Y, VGG_SIZE_Z])

        feed_dict = {image: batch_mini}

        vgg = vgg16.Vgg16()
        with tf.name_scope("content_vgg"):
            vgg.build(image)
	
        act_wanted = [vgg.pool1, vgg.pool2, vgg.pool3, vgg.pool4, vgg.pool5, vgg.fc6, vgg.fc7, vgg.prob]
	act = sess.run(act_wanted, feed_dict=feed_dict)

         
print "completed running the VGG on " + str(batch.shape[0])
print "now saving...."
pickle.dump(act, open( "/tigress/rslee/activations.p", "wb" ) )
print "saved vgg"

# hacking this for now. using two batches instead of one because of memory issues. 
 # smaller batch for testing first
print batch.shape[0]
batch_mini = batch[160:,:,:,:]
print batch_mini.shape[0]




with tf.device('/gpu:0'):
    with tf.Session() as sess:
        image = tf.placeholder("float", [batch_mini.shape[0], VGG_SIZE_X, VGG_SIZE_Y, VGG_SIZE_Z])

        feed_dict = {image: batch_mini}

        vgg = vgg16.Vgg16()
        with tf.name_scope("content_vgg"):
            vgg.build(image)

        act_wanted = [vgg.pool1, vgg.pool2, vgg.pool3, vgg.pool4, vgg.pool5, vgg.fc6, vgg.fc7, vgg.prob]
        act = sess.run(act_wanted, feed_dict=feed_dict)


print "completed running the VGG on " + str(batch.shape[0])
print "now saving...."
pickle.dump(act, open( "/tigress/rslee/activations2.p", "wb" ) )
print "saved vgg"

# for i in xrange(0, batch_mini.shape[0]):
#    utils.print_prob(act[7][i], './synset.txt')       
