from __future__ import division
import numpy as np
from numpy import *
import os
import PIL
from PIL import Image
import base64
import matplotlib
from matplotlib import pylab, mlab, pyplot
%matplotlib inline
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
import pymongo as pm ## first establish ssh tunnel to server where database is running
from matplotlib.path import Path
import matplotlib.patches as patches
from svgpathtools import parse_path

def polyline_pathmaker(lines):
    x = []
    y = []

    codes = [Path.MOVETO] # start with moveto command always
    for i,l in enumerate(lines):
        for _i,_l in enumerate(l):
            x.append(_l[0])
            y.append(_l[1])
            if _i<len(l)-1:
                codes.append(Path.LINETO) # keep pen on page
            else:
                if i != len(lines)-1: # final vertex
                    codes.append(Path.MOVETO)
    verts = zip(x,y)            
    return verts, codes

def path_renderer(verts, codes):
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    if len(verts)>0:
        path = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor='none', lw=2)
        ax.add_patch(patch)
        ax.set_xlim(0,500)
        ax.set_ylim(0,500) 
        ax.axis('off')
        plt.gca().invert_yaxis() # y values increase as you go down in image
        plt.show()
    else:
        ax.set_xlim(0,500)
        ax.set_ylim(0,500)        
        ax.axis('off')
        plt.show()
    plt.savefig()
    plt.close()
    
def flatten(x):
    return [val for sublist in x for val in sublist]


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('save_dir', type=str,default='partial_sketches')
    parser.add_argument('sketch_emb_dir', type=str)
    parser.add_argument('photo_mmemb_dir', type=str)
    args = parser.parse_args()

    num_renders = 24 ## roughly corresponds to the number of TR's in each drawing trial
    sub_paths = [os.path.join('data',i) for i in os.listdir('data') if os.path.isdir(os.path.join('data',i))]
    for s in sub_paths:
        print('printing partial sketches from {}'.format(s))
        X = pd.read_csv(os.path.join(s,s.split('/')[-1] + '_metadata.csv'))
        sub_name = s.split('/')[-1]    
        num_sketches = len(X.svgString.values)
        for sketch_ind in range(num_sketches):
            ## parse path strings only from raw svg dom element
            _X = X.svgString.values[sketch_ind].split('"path":"')
            svg_list = [x.split('","stroke')[0] for x in _X[1:]]

            ## parse into x,y coordinates and output list of lists of coordinates
            lines = []
            Verts = []
            Codes = []
            for stroke_ind,stroke in enumerate(svg_list):
                x = []
                y = []
                parsed = parse_path(stroke)
                for i,p in enumerate(parsed):
                    if i!=len(parsed)-1: # last line segment
                        x.append(p.start.real)
                        y.append(p.start.imag)    
                    else:
                        x.append(p.start.real)
                        y.append(p.start.imag)     
                        x.append(p.end.real)
                        y.append(p.end.imag)
                lines.append(zip(x,y))
                verts, codes = polyline_pathmaker(lines)
                Verts.append(verts)
                Codes.append(codes)

            Verts = flatten(Verts)
            Codes = flatten(Codes)
            splice_markers = map(int,np.linspace(0,len(Verts),num_renders)) 

            for i,t in enumerate(splice_markers[1:]):
                _Verts = Verts[:t]
                _Codes = Codes[:t]            

                ## render and save out image
                fig = plt.figure(figsize=(6,6))
                ax = fig.add_subplot(111)
                if len(verts)>0:
                    path = Path(_Verts, _Codes)
                    patch = patches.PathPatch(path, facecolor='none', lw=5)
                    ax.add_patch(patch)
                    ax.set_xlim(0,500)
                    ax.set_ylim(0,500) 
                    ax.axis('off')
                    plt.gca().invert_yaxis() # y values increase as you go down in image
                    plt.show()
                else:
                    ax.set_xlim(0,500)
                    ax.set_ylim(0,500)        
                    ax.axis('off')
                    plt.show()
                sketch_dir = X.target.values[sketch_ind] + '_' + str(X.trial.values[sketch_ind])
                if not os.path.exists(os.path.join(args.save_dir,sub_name,sketch_dir)):
                    os.makedirs(os.path.join(args.save_dir,sub_name,sketch_dir))
                fpath = os.path.join(args.save_dir,sub_name,sketch_dir,'{}.png'.format(str(i)))
                fig.savefig(fpath)
                plt.close(fig)






