from __future__ import division
import numpy as np
from numpy import *
import os
import PIL
from PIL import Image
import base64
import matplotlib
from matplotlib import pylab, mlab, pyplot
# %matplotlib inline
from IPython.core.pylabtools import figsize, getfigs
plt = pyplot
import seaborn as sns
sns.set_context('poster')
sns.set_style('white')
from matplotlib.path import Path
import matplotlib.patches as patches
from skimage import data, io, filters
import cStringIO
import pandas as pd
import pymongo as pm ## first establish ssh tunnel to Amazon EC2 instance
import cPickle

# jefan 8/28/17
# purpose: extract training phase metadata and sketches produced by participants
# in fMRI training study (aka 'neurosketch')
# dependencies: ssh tunnel to Amazon EC2 instance is open, 
# 'morph_drawing_training_design.pkl' is in current directory,
# and 'get_metadata.py' module is on python path

if __name__ == "__main__":

	# load in experimental design pickle file
	mdtd = cPickle.load(open('morph_drawing_training_design.pkl'))	

	# mongo globals
	conn = pm.MongoClient(port=20809)
	DBNAME = conn['during_morph_drawing_recognition']
	COLNAME = DBNAME['fmri3.files']
	coll=COLNAME
	DATADIR = 'neurosketch_data_3'

	# get worker list
	import get_metadata as gm
	reload(gm)
	workers = gm.get_worker_list(COLNAME)
	print workers
	print 'We have records on ' + str(len(workers)) + ' participants.'	

	# loop through participants and save out metadata and sketches
	for w in workers:
	    try:
	        print 'Now analyzing ' + w + ' ...'
	        # retrieve this participant's records from mongo
	        these = coll.find({'wID': w}).sort('trialNum')   

	        wID = []
	        phase = []
	        version = []
	        category = []
	        viewpoint = []
	        trial = []
	        trialDuration = []
	        target = []
	        competitor = []
	        svgString = []
	        pngString = []

	        # loop through training trials and save relevant metadata
	        for this in these:
	            if this['phase']=='training':
	                wID.append(this['wID'])
	                phase.append(this['phase'])
	                version.append(int(this['versionNum']))
	                category.append(this['category'])
	                viewpoint.append(int(this['viewpoint']))
	                trial.append(int(this['trialNum']))
	                trialDuration.append(float(this['trialDuration']))
	                _target = gm.get_object_name(2,gm.get_object_index(this['morphline'],this['morphnum']))
	                target.append(_target)
	                trained_set = gm.getEndpoints(this['morphline'])
	                _competitor = [i for i in trained_set if i != _target]
	                competitor.append(_competitor[0])
	                svgString.append(this['json'])
	                pngString.append(this['imgData'])
	        ## make pandas dataframe to store metadata
	        X = pd.DataFrame([wID,phase,version,category,viewpoint,trial,trialDuration,
	                         target,competitor,svgString,pngString])
	        X = X.transpose()
	        X.columns = ['wID','phase','version','category','viewpoint','trial','trialDuration',
	                    'target','competitor','svgString','pngString']

	        ## save out dataframe to csv
	        pathdir = os.path.join('data',w)
	        ## save out image as png
	        if not os.path.exists(pathdir):
	            os.makedirs(pathdir)
	        fname = w + '_metadata.csv'
	        filepath = os.path.join(pathdir,fname)
	        X.to_csv(filepath)  
	        ## loop through all sketches and save out as png's in subject specific folders
	        for t in trial:
	            imgData = X[X['trial']==t].pngString.values[0]
	            im = Image.open(cStringIO.StringIO(base64.b64decode(imgData)))
	            fig = plt.figure(figsize=(8,8))
	            p = plt.subplot(1,1,1)
	            plt.imshow(im)
	            k = p.get_xaxis().set_ticklabels([])
	            k = p.get_yaxis().set_ticklabels([])
	            k = p.get_xaxis().set_ticks([])
	            k = p.get_yaxis().set_ticks([])
	            for spine in plt.gca().spines.values():
	                spine.set_visible(False)
	            pathdir = os.path.join('data',X[X['trial']==t].wID.values[0])
	            ## save out image as png
	            if not os.path.exists(pathdir):
	                os.makedirs(pathdir)
	            fname = X[X['trial']==t].wID.values[0]  + '_trial_' + \
	            str(X[X['trial']==t].trial.values[0]) + '_' + X[X['trial']==t].target.values[0]
	            filepath = os.path.join(pathdir,fname)
	            print 'Saving to ' + filepath
	            fig.savefig(filepath+'.png',bbox_inches='tight')  
            	plt.close(fig)
	    except:
	        print('Something went wrong with subject ' + w)	