import numpy as np
import tensorflow as tf

import vgg16
import utils

img1 = utils.load_image("./test_data/limoToSUV_40_15.png.png")

batch1 = img1.reshape((1, 224, 224, 3))



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
       
