import sys
import os
import numpy as np

skip=True
header = ""
all_scores = {}
all_counts = {}
with open(sys.argv[1],'r') as stream:
    for line in stream:
        if skip:
            skip = False
            header=line.strip()
            continue
        line=line.strip().split(',')
        sbjid=line[0]
        scores=[float(s) for s in line[1:]]
        scores = np.array(scores)
        scores = np.log(scores)
        if sbjid in all_scores:
            all_scores[sbjid] = all_scores[sbjid] + scores
            all_counts[sbjid] = all_counts[sbjid] + 1
        else:
            all_scores[sbjid] = scores
            all_counts[sbjid] = 1


print(header)
for k in all_scores:
    all_scores[k] = all_scores[k] / all_counts[k]
    print(k,',',','.join([ str(np.exp(s)) for s in all_scores[k]]),sep='')

    

lab_to_id = {'Angry' : 0 ,'Disgust' : 1,'Fear' : 2,'Happy' : 3,'Neutral' : 4,'Sad' :5 ,'Surprise' : 6}

labels = {}
with open(sys.argv[2],'r') as stream:
    for line in stream:
        [key, label] = line.strip().split()
        label = lab_to_id[label]
        labels[key] = label
        
count = 0
count_corrrect = 0
for k in all_scores:
    pred = np.argmax(all_scores[k])
    if pred == labels[k]:
        count_corrrect += 1
    count += 1
    
print("Accuracy = " , (count_corrrect*1.0)/count)

        
