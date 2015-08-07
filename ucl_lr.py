from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
import sys
import numpy
import time
import theano
import theano.tensor as T
import linecache
import math
import dl_utils as ut
import sys
import time
import random
from time import gmtime, strftime

rng = numpy.random
rng.seed(1234)
numpy.random.seed(10)
one_value = True
k = 3
learning_rate = 0.01
weight_decay = 1E-6

buffer_num = 1000000
													#activation type
train_rounds = 40									 				  	#epochs number
advertiser = '2997'
if len(sys.argv) > 1:
    advertiser = sys.argv[1]
train_file='../../make-ipinyou-data/' + advertiser + '/train.fm.txt' 			#training file
test_file='../../make-ipinyou-data/' + advertiser + '/test.fm.txt'	   			#test file
index_file='../../make-ipinyou-data/' + advertiser + '/featindex.fm.txt'
result_file='../log/ '+ 'log-' + strftime("%Y-%m-%d", gmtime()) + '.txt'

if sys.argv[2]=='mod' and advertiser=='2997':
    train_file=train_file+'.mod4.txt'
#------------------------ 

def sigmoid(p):
    return 1.0 / (1.0 + math.exp(-p))

def pred(x):
    p = w_0
    for (feat, val) in x:
        p += w[feat] * val
    p = sigmoid(p)
    return p

def one_data_y_x(line):
    s = line.strip().replace(':', ' ').split(' ')
    y = int(s[0])
    x = []
    for i in range(1, len(s), 2):
        val = 1
        if not one_value:
            val = float(s[i+1])
        x.append((int(s[i]), val))
    return (y, x)

def pred_best(x,best_w,best_w_0):
    p = best_w_0
    for (feat, val) in x:
        p += best_w[feat] * val
    p = sigmoid(p)
    return p
    
def pred_all(file,best_w,best_w_0):
    fi = open(file, 'r')
    yp = []
    for line in fi:
        data = one_data_y_x(line)
        pclk = pred_best(data[1],best_w,best_w_0)
        yp.append(pclk)
    fi.close()
    print type(yp)
    print len(yp)
    return yp 
    
# initialise
feature_index = {}
index_feature = {}
max_feature_index = 0
feature_num = 0

print 'reading feature index'
fi = open(index_file, 'r')
for line in fi:
    s = line.strip().split('\t')
    index = int(s[1])
    feature_index[s[0]] = index
    index_feature[index] = s[0]
    max_feature_index = max(max_feature_index, index)
fi.close()
feature_num = max_feature_index + 1
print 'feature number: ' + str(feature_num)

print 'initialising'
init_weight = 0.05
w = numpy.zeros(feature_num)
w_0 = 0

# train
best_auc = 0.
best_w_0=0.
bets_w=w
overfitting = False
print 'training:'
fo = open(result_file, 'w')
for round in range(1, train_rounds+1):
    start_time = time.time()
    fi = open(train_file, 'r')
    line_num = 0
    train_data = []
    while True:
        line = fi.readline().strip()
        if len(line) > 0:
            line_num = (line_num + 1) % buffer_num
            train_data.append(one_data_y_x(line))
        if line_num == 0 or len(line) == 0:
            for data in train_data:
                y = data[0]
                x = data[1]
                # train one data
                p = pred(x)
                d = y - p
                w_0 = w_0 * (1 - weight_decay) + learning_rate * d
                for (feat, val) in x:
                    w[feat] = w[feat] * (1 - weight_decay) + learning_rate * d * val
            train_data = []
        if len(line) == 0:
            break
    fi.close()
    train_time = time.time() - start_time
    train_min = int(train_time / 60)
    train_sec = int(train_time % 60)

    # test for this round
    y = []
    yp = []
    fi = open(test_file, 'r')
    for line in fi:
        data = one_data_y_x(line)
        clk = data[0]
        pclk = pred(data[1])
        y.append(clk)
        yp.append(pclk)
    fi.close()
    auc = roc_auc_score(y, yp)
    rmse = math.sqrt(mean_squared_error(y, yp))
    print '%d\t%.8f\t%.8f\t%dm%ds' % (round, auc, rmse, train_min, train_sec)
    fo.write('%d\t%.8f\t%.8f\t%dm%ds\n' % (round, auc, rmse, train_min, train_sec))
    fo.flush()
    if overfitting and auc < best_auc:
        break # stop training when overfitting two rounds already
    if auc > best_auc:
        best_auc = auc
        best_w_0=w_0
        best_w=w
        overfitting = False
    else:
        overfitting = True
       
ut.save_weights("lr_train_"+advertiser+".p",pred_all(train_file,best_w,best_w_0))
ut.save_weights("lr_test_"+advertiser+".p",pred_all(test_file,best_w,best_w_0))
fo.close()


