import numpy as np
import tensorflow as tf

import vgg16
import utils



# to upload multiple images

cars = ['limoToSUV_10','limoToSUV_99','smartToSedan_10','smartToSedan_99'];

batch = np.empty((0, VGG_SIZE_X, VGG_SIZE_Y, VGG_SIZE_Z), float)
for car in cars:  
    for view in xrange(0,NUM_VIEWS):
        imgloc ='https://s3.amazonaws.com/morphrecog-images-1/' + car + '_' + str(view) + '.png.png'
        img = utils.load_image(imgloc)
        img = img.reshape(1, VGG_SIZE_X, VGG_SIZE_Y, VGG_SIZE_Z)
        batch = np.concatenate((batch, img))
        

        
    



# with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
with tf.device('/cpu:0'):
    with tf.Session() as sess:
        image = tf.placeholder("float", [1, 224, 224, 3])

        feed_dict = {image: batch1}

        vgg = vgg16.Vgg16()
        with tf.name_scope("content_vgg"):
            vgg.build(image)

        prob = sess.run(vgg.prob, feed_dict=feed_dict)
        # print(prob)
        utils.print_prob(prob[0], './synset.txt')
       
