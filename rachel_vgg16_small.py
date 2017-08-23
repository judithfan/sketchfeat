# This will be used to send into the gpu. Currently, Tiger is being annoying because it isn't connected to the internet

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
furnitures = ['bedChair_1', 'bedChair_100', 'benchBed_1', 'benchBed_100']


batch = np.empty((0, VGG_SIZE_X, VGG_SIZE_Y, VGG_SIZE_Z), float)
for obj in cars + furnitures:
    for view in xrange(0,NUM_VIEWS):
        imgname = obj + '_' + str(view) + '.png'
        imgloc ='./object_data/' + imgname
        img = utils.load_image(imgloc)
        img = img.reshape(1, VGG_SIZE_X, VGG_SIZE_Y, VGG_SIZE_Z)
        batch = np.concatenate((batch, img))

        
 # smaller batch for testing first
print batch.shape[0]
batch_mini = batch[:4,:,:,:]
print batch_mini.shape[0]




with tf.device('/gpu:0'):
    with tf.Session() as sess:
        image = tf.placeholder("float", [batch_mini.shape[0], VGG_SIZE_X, VGG_SIZE_Y, VGG_SIZE_Z])

        feed_dict = {image: batch_mini}

        vgg = vgg16.Vgg16()
        with tf.name_scope("content_vgg"):
            vgg.build(image)

        prob = sess.run(vgg.prob, feed_dict=feed_dict)
        # print(prob)
        for i in xrange(0, batch_mini.shape[0]):
            utils.print_prob(prob[i], './synset.txt')
        



       
