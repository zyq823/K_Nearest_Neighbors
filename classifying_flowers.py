import numpy as np
import matplotlib.pyplot as plt
import math

# initialization for Question 2

class DataPoint(object): # DataPoint class helps to group data and methods
    def __init__(self, feats):
        self.sLen = feats['sepal_length']
        self.sWid = feats['sepal_width']
        self.pLen = feats['pedal_length']
        self.pWid = feats['pedal_width']
        self.label = feats['class']

    def feature_vector(self):
        return np.array([self.sLen, self.sWid, self.pLen, self.pWid])

    def __str__(self):
        return "Sepal Length: {}, Sepal Width: {}, Pedal Length: {}, Pedal Width: {}, Class: {}".format(self.sLen, self.sWid, self.pLen, self.pWid, self.label)
        
# the following code is for Question 2.a.i

def parse_dataset(filename):
    data_file = open(filename, 'r')  # Open File "to read"
    dataset = []  # List to hold Datapoint objects
    label_map = {'Iris-setosa': 1, 'Iris-versicolor': 2, "Iris-virginica": 3}

    for index, line in enumerate(data_file):
        if index == 0:  # First line describes the datapoint, it's not an actual datapoint, 
            continue  # do nothing, it will skip all the following code
        sLen, sWid, pLen, pWid, label = line.strip().split(',')  # strip() removes '\n', and split(',') splits the line at tabs
        dataset.append(DataPoint({'sepal_length':float(sLen), 'sepal_width':float(sWid), 'pedal_length':float(pLen), 'pedal_width':float(pWid), 'class':label_map[label]}))  # Create DataPoint object for the given data

    print("Total Number of Data Points: {0}".format(len(dataset)))
    print("Number of Data Points by Iris Type: Iris-setosa - {0} , Iris-versicolor - {1} , Iris-virginica - {2}".format(len([i for i in dataset if i.label == 1]), len([i for i in dataset if i.label == 2]), len([i for i in dataset if i.label == 3])))
    return dataset

train_set = parse_dataset('iris_train.csv')

# the following code is for Question 2.a.iI

def plot_histograms(dataset):
    s_lengths = [data.sLen for data in dataset]
    s_widths = [data.sWid for data in dataset]
    p_lengths = [data.pLen for data in dataset]
    p_widths = [data.pWid for data in dataset]
    
    plt.figure(figsize = (10, 10))
    
    h1 = plt.subplot(221)
    h1.hist(s_lengths, bins = 10)
    plt.title('Sepal Length')

    h2 = plt.subplot(222)
    h2.hist(s_widths, bins = 10)
    plt.title('Sepal Width')
    
    h3 = plt.subplot(223)
    h3.hist(p_lengths, bins = 10)
    plt.title('Pedal Length')
   
    h4 = plt.subplot(224)
    h4.hist(p_widths, bins = 10)
    plt.title('Pedal Width')
    
    plt.suptitle('Histogram for Iris Features')
    plt.show()

# plot_histograms(train_set)

# the following code is for Question 2.a.iii

def plot_scatters(dataset):
    s_len1 = [data.sLen for data in dataset if data.label == 1]
    s_wid1 = [data.sWid for data in dataset if data.label == 1]
    p_len1 = [data.pLen for data in dataset if data.label == 1]
    p_wid1 = [data.pWid for data in dataset if data.label == 1]

    s_len2 = [data.sLen for data in dataset if data.label == 2]
    s_wid2 = [data.sWid for data in dataset if data.label == 2]
    p_len2 = [data.pLen for data in dataset if data.label == 2]
    p_wid2 = [data.pWid for data in dataset if data.label == 2]

    s_len3 = [data.sLen for data in dataset if data.label == 3]
    s_wid3 = [data.sWid for data in dataset if data.label == 3]
    p_len3 = [data.pLen for data in dataset if data.label == 3]
    p_wid3 = [data.pWid for data in dataset if data.label == 3]

    plt.figure(figsize = (9, 6))
    
    s1 = plt.subplot(231)
    s1.scatter(s_len1, s_wid1, c='b', marker='x', label='setosa')
    s1.scatter(s_len2, s_wid2, c='r', marker='o', label='versicolor')
    s1.scatter(s_len3, s_wid3, c='g', marker='+', label='virginica')
    plt.xlabel('sepal width')
    plt.ylabel('sepal length')
    plt.legend()

    s2 = plt.subplot(232)
    s2.scatter(s_len1, p_len1, c='b', marker='x', label='setosa')
    s2.scatter(s_len2, p_len2, c='r', marker='o', label='versicolor')
    s2.scatter(s_len3, p_len3, c='g', marker='+', label='virginica')
    plt.xlabel('pedal length')
    plt.ylabel('sepal length')
    plt.legend()

    s3 = plt.subplot(233)
    s3.scatter(s_len1, p_wid1, c='b', marker='x', label='setosa')
    s3.scatter(s_len2, p_wid2, c='r', marker='o', label='versicolor')
    s3.scatter(s_len3, p_wid3, c='g', marker='+', label='virginica')
    plt.xlabel('pedal width')
    plt.ylabel('sepal length')
    plt.legend()

    s4 = plt.subplot(234)
    s4.scatter(s_wid1, p_len1, c='b', marker='x', label='setosa')
    s4.scatter(s_wid2, p_len2, c='r', marker='o', label='versicolor')
    s4.scatter(s_wid3, p_len3, c='g', marker='+', label='virginica')
    plt.xlabel('pedal length')
    plt.ylabel('sepal width')
    plt.legend()

    s5 = plt.subplot(235)
    s5.scatter(s_wid1, p_wid1, c='b', marker='x', label='setosa')
    s5.scatter(s_wid2, p_wid2, c='r', marker='o', label='versicolor')
    s5.scatter(s_wid3, p_wid3, c='g', marker='+', label='virginica')
    plt.xlabel('pedal width')
    plt.ylabel('sepal width')
    plt.legend()

    s6 = plt.subplot(236)
    s6.scatter(p_len1, p_wid1, c='b', marker='x', label='setosa')
    s6.scatter(p_len2, p_wid2, c='r', marker='o', label='versicolor')
    s6.scatter(p_len3, p_wid3, c='g', marker='+', label='virginica')
    plt.xlabel('pedal width')
    plt.ylabel('pedal length')
    plt.legend()

    plt.suptitle('Scatter Plots for Iris Features')
    plt.show()

# plot_scatters(train_set)

# the following code is for Question 2.b.i

def l2_norm(iris1, iris2):
    """
    Calculate euclidean distance
    """
    distance = 0.0
    i1_feats = iris1.feature_vector()
    i2_feats = iris2.feature_vector()
    for i in range(len(i1_feats)):
        distance += (i1_feats[i]-i2_feats[i])**2
    return math.sqrt(distance)

def k_nearest_neighbors(train_set, dp, k_value):
    """
    Find k nearest neighbors
    """
    distances = []
    for t in train_set:
        distances.append((t, l2_norm(dp, t)))
    distances.sort(key=lambda tup: tup[1])
    neighbors = distances[:k_value]
    count1, count2, count3 = 0, 0, 0
    for n in neighbors:
        iris = n[0]
        if iris.label == 1:
            count1 +=1
        elif iris.label == 2:
            count2 +=1
        else:
            count3 +=1
    iris_types = [count1, count2, count3]
    return iris_types.index(max(iris_types))+1

dev_set = parse_dataset('iris_dev.csv')
k_values = np.arange(1, 20, 2).tolist()

# the following code is for Question 2.b.ii

def acc_metric(train_set, the_set, k_values):
    set_size = len(the_set)
    accuracy = []
    for k in k_values: 
        count = 0
        for dp in the_set:
            correct = dp.label
            if k_nearest_neighbors(train_set, dp, k) == correct:
                count +=1
        accuracy.append(count / set_size)
    if len(k_values) == 1: # this applies to Question 2.b.iii
        return accuracy[0]
    else:
        # plot_acc(accuracy, k_values)
        return k_values[accuracy.index(max(accuracy))]

def plot_acc(acc, k_values):
    plt.figure(figsize = (10, 10))
    plt.plot(k_values, acc)
    plt.title('Acc Metric on Dev Set')
    plt.xlabel('k-value')
    plt.ylabel('classification accuracy')
    plt.show()
    

best_k = acc_metric(train_set, dev_set, k_values)

# the following code is for Question 2.b.iii

test_set = parse_dataset('iris_test.csv')
best_k_value = [best_k]
acc_test = acc_metric(train_set, test_set, k_values)

# the following code is for Question 2.b.iv

def cos_sim(iris1, iris2):
    """
    Calculate cosine similarity
    """
    numerator, i1_denom, i2_denom = 0.0, 0.0, 0.0
    i1_feats = iris1.feature_vector()
    i2_feats = iris2.feature_vector()
    for i in range(len(i1_feats)):
        numerator += (i1_feats[i] * i2_feats[i])
        i1_denom += i1_feats[i]**2
        i2_denom += i2_feats[i]**2
    denominator = math.sqrt(i1_denom) * math.sqrt(i2_denom)
    return math.degrees(math.acos(numerator / denominator))

def knn_cos_sim(train_set, dp, k_value):
    """
    Find k nearest neighbors
    """
    distances = []
    for t in train_set:
        distances.append((t, cos_sim(dp, t)))
    distances.sort(key=lambda tup: tup[1])
    neighbors = distances[:k_value]
    count1, count2, count3 = 0, 0, 0
    for n in neighbors:
        iris = n[0]
        if iris.label == 1:
            count1 +=1
        elif iris.label == 2:
            count2 +=1
        else:
            count3 +=1
    iris_types = [count1, count2, count3]
    return iris_types.index(max(iris_types))+1

def acc_cos_sim(train_set, the_set, k_values):
    set_size = len(the_set)
    accuracy = []
    for k in k_values: 
        count = 0
        for dp in the_set:
            correct = dp.label
            if knn_cos_sim(train_set, dp, k) == correct:
                count +=1
        accuracy.append(count / set_size)
    if len(k_values) == 1: # this applies to Question 2.b.iii
        return accuracy[0]
    else:
        # plot_acc(accuracy, k_values)
        return k_values[accuracy.index(max(accuracy))]

best_k_cs = acc_cos_sim(train_set, dev_set, k_values)
best_k_cs_value = best_k_cs
acc_test_cs = acc_cos_sim(train_set, test_set, best_k_cs_value)