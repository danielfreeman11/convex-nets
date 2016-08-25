import os
import numpy as np

data = []

for f in os.listdir("."):
	if len(f.split("."))==3 and f.split(".")[1]=='o38675' and int(f.split(".")[2])<1000:
		with open(f) as file:
			lines = file.readlines()
			if len(lines) > 2 and len(lines) < 7:
				data.append([float(p) for p in [lines[0].split(" ")[1], lines[1].split(" ")[1], lines[3], lines[4]]])


#print data

#for d in data:
#	print str(d[0]) + "\t" + str(d[1]) + "\t" + str(d[2]) + "\t" + str(d[3])


dictcheck = []
mydict = {}

for d in data:
	if d[0] not in dictcheck:
		dictcheck.append(d[0])
		mydict[d[0]] = []

for d in data:
	mydict[d[0]].append(d[1])

for d in dictcheck:
	print str(d) + "\t" + str(np.mean(mydict[d])) + "\t" + str(np.std(mydict[d])/np.sqrt(len(mydict[d])))

