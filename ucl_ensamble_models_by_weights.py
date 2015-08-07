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
import ucl_gaussian_binary_rbm as gbrbm
import sys
import time
import random
from time import gmtime, strftime
import pickle
import ucl_mlp3fm as mlpfm

rng = numpy.random
rng.seed(1234)
numpy.random.seed(10)
one_value = True
learning_rate = 0.01
weight_decay = 0#1E-6
buffer_num = 1000000											#activation type
train_rounds = 40									 				  	#epochs number
advertiser = '2997'
if len(sys.argv) > 1:
    advertiser = sys.argv[1]
train_file='../../make-ipinyou-data/' + advertiser + '/train.fm.txt' 			#training file
test_file='../../make-ipinyou-data/' + advertiser + '/test.fm.txt'	   			#test file
index_file='../../make-ipinyou-data/' + advertiser + '/featindex.fm.txt'
result_file='../log/ '+ 'log-' + strftime("%Y-%m-%d", gmtime()) + '.txt'

feature_num = 3

# --------------------------------------- 

mfm=mlpfm.MLP_FM()
def sigmoid(p):
    return 1.0 / (1.0 + math.exp(-p))
def get_new_x(x):
    new_x=[]
    new_x.append(pred_lr(x)) 
    new_x.append(pred_fm(x))
    new_x.append(mfm.get_pred(x)[0])
    
    return new_x
def pred(x):
    p = w_0
    p = ((p+numpy.dot(w,get_new_x(x))))
#     print 'p',p
    return p

def pred_lr(x):
    p = b1
    for (feat, val) in x:
        p += w1[feat] * val
    p = sigmoid(p)
    return p
    
def pred_fm(x):
    p = b2
    sum_1 = 0
    sum_2 = 0
    for (feat, val) in x:
        tmp = v2[feat] * val
        sum_1 += tmp
        sum_2 += tmp * tmp
    p = numpy.sum(sum_1 * sum_1 - sum_2) / 2.0 + b2
    for (feat, val) in x:
        p += w2[feat] * val
    p = sigmoid(p)
    return p
def pred_mlpfm(x):
    return mlpfm.get_pred(x)
    
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



# initialise
(w1,b1)=pickle.load(open( "lr_"+advertiser+".p", "rb" ) )
(w2,v2,b2)=pickle.load(open( "fm_"+advertiser+".p", "rb" ) )

feature_index = {}
index_feature = {}
max_feature_index = 0


print 'reading feature index'
fi = open(index_file, 'r')
for line in fi:
    s = line.strip().split('\t')
    index = int(s[1])
    feature_index[s[0]] = index
    index_feature[index] = s[0]
    max_feature_index = max(max_feature_index, index)
fi.close()
# feature_num = max_feature_index + 1
print 'feature number: ' + str(feature_num)

print 'initialising'
init_weight = 0.05
w = numpy.zeros(feature_num)
w_0 = 0



# init weight test for this round
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
print 'init auc', auc



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
                grad=learning_rate * d * numpy.asarray(get_new_x(x))
                w = w * (1 - weight_decay) + grad
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
    print 'emsamble:test %d\t%.8f\t%.8f\t%dm%ds' % (round, auc, rmse, train_min, train_sec)
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
# ut.save_weights("lr_"+advertiser+"_.p",(best_w,best_w_0))
fo.close()

