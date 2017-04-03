import config as cfg
import utils
from os import listdir
from os.path import isfile, join


compound = []
for fName in [f for f in listdir(cfg.PATH_OUTPUT) if (isfile(join(cfg.PATH_OUTPUT, f)) & (f.endswith('.txt')))]:
	results = dict()
	title = ''
	f = ''
	fstd = ''
	mae = ''
	maestd = ''
	for line in open(cfg.PATH_OUTPUT+fName,'r'):
		line = line.replace('\n','')
		if ('Executing for' in line):
			#write away previous results
			results[title] = [mae, maestd, f, fstd]
			
			title = 'cross_' + utils.find_between(line, 'Executing for ', 'model').strip()
		elif('Average Mean absolute error (official): ' in line):
			mae = line.split("(official): ",1)[1] 
		elif('Average Mean absolute error (official); std: ' in line):
			maestd = line.split('std: ',1)[1]
		elif('Average F1-measure: ' in line):
			f = line.split("measure: ",1)[1]
		elif('Average F1-measure; std: ' in line):
			fstd = line.split("std: ",1)[1]
		elif("Scores for " in line):
			#write away previous, doing for test now
			results[title] = [mae, maestd, f, fstd]
			
			title = 'test_' + utils.find_between(line, 'Scores for ', ':').strip()
	results[title] = [mae, maestd, f, fstd]
	compound.append([fName, results])

for line in compound:
	print(',Development phase (10-fold CV), ,Testing phase, ')
	print('System, MAE, F-score, MAE, F-score')
	rows = []
	for key in line[1]:
		if ('cross' in key):
			key = key.replace('cross_','')
			row = key + '+' + line[0] + ', '
			#get cross val for this setup
			vals = line[1]['cross_'+key]
			try:
				row += str(round(float(vals[0]),2)) + " (SD "+str(round(float(vals[1]),2))+"), "+ str(round(float(vals[2])*100,2)) + " (SD "+str(round(float(vals[3])*100,2))+"), "
			except:
				row += vals[0] + " (SD "+vals[1]+"), "+ vals[2] + " (SD "+vals[3]+"), "
			#get test val for this setup
			#vals = line[1]['test_'+key]
			try:
				vals = line[1]['test_'+key]
			except:
				vals = ['N/A','N/A','N/A','N/A']
				
			try:
				row += str(round(float(vals[0]),2)) + "," + str(round(float(vals[2])*100,2))
			except:
				row += vals[0] + "," + vals[2]
			rows.append(row)
	rows = sorted(rows)
	print('\n'.join(rows))

