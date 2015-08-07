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

rng = numpy.random
rng.seed(1234)
numpy.random.seed(10)
one_value = True
learning_rate = 1
weight_decay = 1E-10
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


def sigmoid(p):
    return 1.0 / (1.0 + math.exp(-p))

def pred(x):
    p = w_0
    p = ((p+numpy.dot(w,x)))
    p=sigmoid(p)
    return p

    
def one_data_y_x(file,line,index):
    s = line.strip().replace(':', ' ').split(' ')
    y = int(s[0])
    x = []
    if file=='train':
        x.append(lr_pred[index])
        x.append(fm_pred[index])
        x.append(mlp3fm_pred[index])
    else:
        x.append(lr_test[index])
        x.append(fm_test[index])
        x.append(mlp3fm_test[index])
    return (y, x)

# initialise
lr_pred=pickle.load(open( "lr_train_"+advertiser+".p", "rb" ) )
lr_test=pickle.load(open( "lr_test_"+advertiser+".p", "rb" ) )
fm_pred=pickle.load(open( "fm_train_"+advertiser+".p", "rb" ) )
fm_test=pickle.load(open( "fm_test_"+advertiser+".p", "rb" ) )
mlp3fm_pred=pickle.load(open( "mlp3fm_train_"+advertiser+".p", "rb" ) )
mlp3fm_test=pickle.load(open( "mlp3fm_test_"+advertiser+".p", "rb" ) )
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
print 'learing rate: ',learning_rate
print 'weight_decay: ',weight_decay
print 'initialising'

# train
w = numpy.zeros(feature_num)
w_0 = 0
best_auc = 0.
best_w_0=0.
bets_w=w
overfitting = 0
print 'training:'
fo = open(result_file, 'w')
for round in range(1, train_rounds+1):
    start_time = time.time()
    fi = open(train_file, 'r')
    line_num = 0
    train_data = []
    index=0
    while True:
        line = fi.readline().strip()
        if len(line) > 0:
            line_num = (line_num + 1) % buffer_num
            train_data.append(one_data_y_x('train',line,index))
            index+=1
        if line_num == 0 or len(line) == 0:
            for data in train_data:
                y = data[0]
                x = data[1]
                # train one data
                p = pred(x)
                d = y - p
                w_0 = w_0 * (1 - weight_decay) + learning_rate * d
                grad=learning_rate * d * numpy.asarray(x)
                w = w * (1 - weight_decay) + grad
            train_data = []
        if len(line) == 0:
            break
    fi.close()
    train_time = time.time() - start_time
    train_min = int(train_time / 60)
    train_sec = int(train_time % 60)




 # train error for this round
    y_train=[]
    yp_train=[]
    fi = open(train_file, 'r')
    index=0
    for line in fi:
        data_train = one_data_y_x('train',line,index)
        index+=1
        clk_train = data_train[0]
        pclk_train = pred(data_train[1])
        y_train.append(clk_train)
        yp_train.append(pclk_train)
    fi.close()
    auc_train = roc_auc_score(y_train, yp_train)
    rmse_train = math.sqrt(mean_squared_error(y_train, yp_train))
    print 'ensamble: \t %d  train:%.8f\t%.8f\t%dm%ds' % (round,auc_train,rmse_train, train_min, train_sec)
    fi.close()
    
    # test for this round
    y = []
    yp = []
    fi = open(test_file, 'r')
    index=0
    for line in fi:
        data = one_data_y_x('test',line,index)
        index+=1
        clk = data[0]
        pclk = pred(data[1])
        y.append(clk)
        yp.append(pclk)
    fi.close()
    auc = roc_auc_score(y, yp)
    rmse = math.sqrt(mean_squared_error(y, yp))
    print 'ensamble:test %d  %.8f\t%.8f\t%dm%ds' % (round, auc, rmse, train_min, train_sec)
    fo.write('%d\t%.8f\t%.8f\t%dm%ds\n' % (round, auc, rmse, train_min, train_sec))
    fo.flush()
    if overfitting>1 and auc < best_auc:
        break # stop training when overfitting two rounds already
    if auc > best_auc:
        best_auc = auc
        best_w_0=w_0
        best_w=w
        overfitting -=1
    else:
        overfitting +=1
print 'best auc',best_auc
# ut.save_weights("lr_"+advertiser+"_.p",(best_w,best_w_0))
fo.close()

