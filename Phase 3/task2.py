import numpy as np  
import csv 
import pandas as pd  
from collections import Counter
from collections import defaultdict
from sklearn.metrics import accuracy_score
import json
 
choice = input('Choose a classification algorithm of your choice \n 1. KNN based classifier 2. PPR based classifier :')

def train(X_train, y_train):
	return

def predict(X_train, y_train, x_test, k):
	# create list for distances and targets
	distances = []
	targets = []

	for i in range(len(X_train)):
		# first we compute the euclidean distance
		distance = np.sqrt(np.sum(np.square(pd.to_numeric(x_test) - pd.to_numeric(X_train.iloc[i, :]))))
		# add it to list of distances
		distances.append([distance, i])
        
	# sort the list
	distances = sorted(distances)
	# make a list of the k neighbors' targets
	for i in range(k):
		index = distances[i][1]
		targets.append(y_train[index])

	# return most common target
	return Counter(targets).most_common(1)[0][0]

def kNearestNeighbor(X_train, y_train, X_test, predictions, k):
	# check if k larger than n
	if k > len(X_train):
		raise ValueError
		
	# train on the input data
	train(X_train, y_train)

	# predict for each testing observation
	for i in range(len(X_test)):
		predictions.append(predict(X_train, y_train, X_test.iloc[i, :], k))


top_k_values = json.load(open("./top_k_values.json"))
top_k_files = json.load(open("./top_k_files.json"))

gesture_fn = list()
for key in top_k_values:
    gesture_fn.append(key)
 
#Return the PageRank of the nodes in the graph. 
#PageRank computes a ranking of the nodes 
#in the graph G based on the structure of the incoming links.    
def pagerank(Graph_comp_1, Graph_comp_2, alpha=0.85, personalization=None, 
			max_iter=100, tol=1.0e-6, nstart=None, 
			dangling=None): 

    if len(Graph_comp_1) == 0: 
        return {} 

    N = len(Graph_comp_1)

    # Choose fixed starting vector if not given 
    if nstart is None: 
        x = dict.fromkeys(list(Graph_comp_1.keys()), 1.0 / N)
    else:
        # Normalized nstart vector 
        s = float(sum(nstart.values())) 
        x = dict((k, v / s) for k, v in nstart.items()) 
    if personalization is None: 
        # Assign uniform personalization vector if not given 
        p = dict.fromkeys(list(Graph_comp_1.keys()), 1.0 / N)
    else: 
        s = float(sum(personalization.values()))
        p = dict((k, v / s) for k, v in personalization.items())
    i = 0
    for _ in range(max_iter):
        i += 1
        xlast = x
        x = dict.fromkeys(xlast.keys(), 0) 
    
        for n in x:
            temp_sum = sum(b for b in Graph_comp_2[n])
            for fn,dis in zip(Graph_comp_1[n],Graph_comp_2[n]):
                x[fn] += (xlast[n]/len(Graph_comp_1[n])) * (dis / temp_sum)
        for n in x:
            x[n] = x[n] * alpha + (1 - alpha) * p[n]
		# check convergence, l1 norm 
        err = sum([abs(x[n] - xlast[n]) for n in x])
        if err < N*tol: 
#            print("Converged at ", i)
            return x
        
        
if(choice == '1'):
    
    train_label = []
    
    # read the input sample training labels file
    sample = pd.read_excel("sample_training_labels.xlsx", header = None)
    
    train_label = []
    for index , row in sample.iterrows():
        train_label.append(list(row))
     
    
    # load the TF vector values
    csvfile = open("./all_tfidf_gestures.csv",
                   "r", encoding="utf8")
    
    #csvfile = csvfile.iloc[1:,:]
    
    reader = csv.reader(csvfile, delimiter=',', quotechar = "'")
    list_of_lines = list(reader)
    
    loader_list = list(map(list, zip(*list_of_lines)))
    
    # Create a DataFrame to hold the TF vector values
    df = pd.DataFrame(loader_list)
    df.columns = df.iloc[0]
    df = df[1:]
    df = df.iloc[:,1:]
    
    x_train = []
    y_train = []
    x_test = []
    train_gesture = []
    
    gest = []
    for pair in train_label:
        temp = str(pair[0])
        temp = temp + '.wrd'
        gest.append(temp)
        
    for i in range(len(train_label)):
        train_gesture.append(train_label[i][0])
        x_train.append(list(df[gest[i]]))
        y_train.append(train_label[i][1])
    
    # remove the sample gesture files from the test gesture files list 
    test_gesture = list(set(df.columns)-set(train_gesture))
    
    for column in test_gesture:
        x_test.append(list(df[column]))
    
    # Convert the list into DataFrame
    x_test = pd.DataFrame(x_test)
    x_train = pd.DataFrame(x_train)
        
    predictions = []
    try:
    	# call the k-nearest classifier
        kNearestNeighbor(x_train, y_train, x_test, predictions, 20)
    
        # store the predictions under y_pred
        y_pred = np.asarray(predictions)
    except ValueError:
        print("Can't have more neighbors than training samples!!")
    
    
    results = dict()
    x_test_img = np.array(test_gesture).tolist()
    
    for img, label in (zip(x_test_img, y_pred)):
    	# store the labels under corresponding gesture files
        results[img] = label
    
    #print(results)
    
    sample_all = pd.read_excel("all_labels.xlsx", header = None)
    
    des_res = []
    for index , row in sample_all.iterrows():
        des_res.append(list(row))
    
    des_lab = []
    file_no = []
    for pair in des_res:
        des_lab.append(pair[1])
        file_no.append(pair[0])
    
    #pred_lab = []
    #for i,v in results.items():
    #    pred_lab.append(v)
        
    fin_res = []
    pred_lab = []
    for i,key in zip(file_no,results):
        key =  str(i) + '.wrd'
        temp = [key,results[key]]
        fin_res.append(temp)
        pred_lab.append(results[key])
    
    for ele in fin_res:
        print(ele)
        
    accuracy_score = (accuracy_score(pred_lab, des_lab))*100
    accuracy_score = round(accuracy_score,2)
    print(f'Accuracy: {accuracy_score} %')

else:
    # read the input from the text file
    sample = pd.read_excel("sample_training_labels.xlsx", header = None)
    
    train_label = []
    for index , row in sample.iterrows():
        train_label.append(list(row))
    
    x_train = []
    y_train = []
    x_test = []
    train_gesture = []
    
    for i in range(len(train_label)):
        train_gesture.append(train_label[i][0])
        y_train.append(train_label[i][1])
    
    # create a distinct labels set     
    labels = list(set(y_train))
    
    label_to_gesture = defaultdict(list)
    
    scores = dict()
    
    for gesture, label in zip(train_gesture, y_train):
    	# assign the label to the gesture file name
        label_to_gesture[label].append(gesture)
    
    cont = 'y'
    for label in labels:
        ges = label_to_gesture[label]
        gest = []
        for ele in ges:
            ele = str(ele)
            ele = ele +'.wrd'
            gest.append(ele)
            
        person_dict = dict.fromkeys(gesture_fn, 0.0)
        
        for gesture in gest:
            if gesture in person_dict:
                person_dict[gesture] = 1.0
            else:
                print("Gesture {} not found".format(gesture))
    
        # call the pagerank function 
        ppr = pagerank(top_k_files, top_k_values, 0.85, personalization=person_dict, max_iter=100)
        sorted_by_val = sorted(ppr.items(), key=lambda kv: kv[1], reverse = True)
        for g in sorted_by_val:
            if g[0] == '30':
                print(gest, g[0], g[1], label)
            scores[(g[0],label)] = g[1]
    
    # create a default dictionary to store the gesture labels
    gesture_label = defaultdict(list)
    for gesture in gesture_fn:
        temp_score = list()
        for label in labels:
            temp_score.append([gesture, label, scores[(gesture, label)]])
        temp_score = sorted(temp_score, key=lambda x:x[2], reverse=True)
        gesture_label[gesture].append([temp_score[0][1],temp_score[0][2]])
        i = 1
        while i < len(labels) and temp_score[i][2] == gesture_label[gesture][0][1]:
            gesture_label[gesture].append([temp_score[i][1],temp_score[i][2]])
            i += 1
    
    # remove the sample gesture files from the test gesture files list  
    test_gesture = list(set(gesture_fn)-set(train_gesture))
    #print(gesture_label)
    
    sample_all = pd.read_excel("all_labels.xlsx", header = None)
    
    des_res = []
    for index , row in sample_all.iterrows():
        des_res.append(list(row))
    
    des_lab = []
    file_no = []
    for pair in des_res:
        des_lab.append(pair[1])
        file_no.append(pair[0])
    
    results = []    
    pred_lab = []
    for i,key in zip(file_no,gesture_label):
        key = str(i) + '.wrd'
        lab = gesture_label[key][0][0]
        temp = [key,lab]
        results.append(temp)
        pred_lab.append(gesture_label[key][0][0])
    
    for ele in results:
        print(ele)
    
    accuracy_score = (accuracy_score(pred_lab, des_lab))*100
    accuracy_score = round(accuracy_score,2)
    print(f'Accuracy: {accuracy_score} %')
    
