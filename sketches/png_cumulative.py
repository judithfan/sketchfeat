### Put script in sketches folder
### For each subject, if the metadata csv is found: for each stroke in each trial, render path, write to actual svg file, convert to png, resize png



### REMAINING ISSUES: There is probably a more elegant way to go straight from the svg path attributes in the metadata to a resized png. This is kind of jury rigged, but functional
### some participants only use a stroke or two, should we aim for finer time scale?
### some images get a weird artifact, where there is a line drawn directly to the border of the image (not sure why)



import pandas as pd
import os
from PIL import Image
from svgpathtools import wsvg, parse_path
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM


currentdir = os.getcwd()
sketchdir = str(currentdir) + '/sketch_data'

dims = (224,224)
atts = {'fill': 'none', 'stroke': '#000000', 'stroke-width':'5'}
frame = 'M0,0L0,460L460,460L460,0'
frameatts = {'fill': 'none', 'stroke': '#ffffff', 'stroke-width':'0.01'}

#For testing
#subjects = ['0110171_neurosketch','1121161_neurosketch']

subjects = ['1121161_neurosketch','1130161_neurosketch','1201161_neurosketch','1202161_neurosketch','1203161_neurosketch','1206161_neurosketch','1206162_neurosketch','1206163_neurosketch','1207161_neurosketch','1207162_neurosketch','1207162_neurosketch','0110171_neurosketch','0110172_neurosketch', '0111171_neurosketch', '0112171_neurosketch', '0112172_neurosketch', '0112173_neurosketch', '0113171_neurosketch','0115172_neurosketch', '0115174_neurosketch','0117171_neurosketch', '0118171_neurosketch','0118172_neurosketch', '0119171_neurosketch', '0119172_neurosketch', '0119173_neurosketch', '0119174_neurosketch', '0120171_neurosketch', '0120172_neurosketch', '0120173_neurosketch', '0123171_neurosketch', '0123172_neurosketch', '0123173_neurosketch',  '0124171_neurosketch', '0125171_neurosketch', '0125172_neurosketch']



for subject in subjects:
	dir = str(sketchdir) + '/' + str(subject) + '_metadata.csv'
	outputdir = str(sketchdir) + '/' + str(subject)
	if os.path.exists(dir):
		if not os.path.exists(outputdir):
			os.makedirs(outputdir)
		metadata = pd.read_csv(str(dir), delimiter = ',' , quotechar = '"')
		numTrials = len(metadata.index)
		for trial in range(0,numTrials):
			
			strokes = [parse_path(frame)]
			attlist = [frameatts]

			n = 0
			trial_num = metadata.loc[trial, 'trial']
			item = metadata.loc[trial, 'target']
			svgdata = metadata.loc[trial,'svgString']

			paths = svgdata.split("M")
			paths = paths[1:]
			
			for path in paths:
				
				outfile = str(outputdir) + "/" + str(subject) + '_trial_' + str(trial_num) + '_' + str(item) + str(n)

				subdiv_path = path.split('"')
				path_no_M = subdiv_path[0]
				new_path = "M" + str(path_no_M)
				stroke = parse_path(new_path)
				strokes.append(stroke)
				attlist.append(atts)
				
				wsvg(strokes, attributes=attlist, filename = str(outfile) + '.svg')
				interim = svg2rlg(str(outfile) + '.svg')
				renderPM.drawToFile(interim, str(outfile + '.png'))
				interim2 = Image.open(str(outfile + '.png'))
				interim2.resize(dims)
				interim2.save(str(outfile) + '.png')
				os.remove(str(outfile) + '.svg')
				
				n = n + 1
		
		print('Subject complete: ' + str(subject))
	else:
		print('MISSING FILE: ' + str(subject) + '_metadata.csv !!')
		
		


	
