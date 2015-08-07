#!/usr/bin/python
import sys
import time
import math
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
from time import gmtime, strftime
import dl_utils as ut
advertiser = '2997'
if len(sys.argv) > 1:
    advertiser = sys.argv[1]
train_file='../../make-ipinyou-data/' + advertiser + '/train.fm.txt'             #training file
test_file='../../make-ipinyou-data/' + advertiser + '/test.fm.txt'                   #test file
index_file='../../make-ipinyou-data/' + advertiser + '/featindex.fm.txt'
result_file='../log/ '+ 'log-fm-' + strftime("%Y-%m-%d", gmtime()) + '.txt'
model_file='../../make-ipinyou-data/' + advertiser + '/fm.model.txt'
trainlog_file='../../make-ipinyou-data/' + advertiser + '/train.log.txt'             #training file
testlog_file='../../make-ipinyou-data/' + advertiser + '/test.log.txt'
traindl_file='../../make-ipinyou-data/' + advertiser + '/train.dl.txt'             #training file
testdl_file='../../make-ipinyou-data/' + advertiser + '/test.dl.txt'    
def sigmoid(p):
    return 1.0 / (1.0 + math.exp(-p))

def pred_lr(x):
    p = w_0
    for (feat, val) in x:
        p += w[feat] * val
    p = sigmoid(p)
    return p

def pred(x):
    p = w_0
    sum_1 = 0
    sum_2 = 0
    for (feat, val) in x:
        tmp = v[feat] * val
        sum_1 += tmp
        sum_2 += tmp * tmp
    p = np.sum(sum_1 * sum_1 - sum_2) / 2.0 + w_0
    for (feat, val) in x:
        p += w[feat] * val
    p = sigmoid(p)
    return (p, sum_1)

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

def output_model(model_file):
    print 'output model to ' + model_file
    foo = open(model_file, 'w')
    foo.write('%.5f %d %d\n' % (w_0, feature_num, k))
    for i in range(feature_num):
        foo.write('%d %.5f' % (i, w[i]))
        for j in range(k):
            foo.write(' %.5f' % v[i][j])
        foo.write(' %s\n' % index_feature[i])
    foo.close()

def load_model(model_file):
    global feature_num, k, w_0, w, v, index_feature, feature_index
    print 'loading model from ' + model_file
    fi = open(model_file, 'r')
    line_num = 0
    for line in fi:
        line_num += 1
        s = line.strip().split()
        if line_num == 1:
            w_0 = float(s[0])
            feature_num = int(s[1])
            k = int(s[2])
            v = np.zeros((feature_num, k))
            w = np.zeros(feature_num)
            index_feature = {}
            feature_index = {}
        else:
            i = int(s[0])
            w[i] = float(s[1])
            for j in range(2, 2 + k):
                v[i][j] = float(s[j])
            feature = s[2 + k]
            index_feature[i] = feature
            feature_index[feature] = i
    fi.close()


feature_cols = ['weekday', 'hour', 'useragent', 'IP', 'region', 'city', 'adexchange', 'domain', 'slotid',
       'slotwidth', 'slotheight', 'slotvisibility', 'slotformat', 'creative', 'advertiser'] #, 'usertag']

feature_cols_special = ['slotprice']

def feat_trans(name, content):
    content = content.lower()
    if name == 'slotprice':
        price = int(content)
        if price > 100:
            return '101+'
        elif price > 50:
            return '51-100'
        elif price > 10:
            return '11-50'
        elif price > 0:
            return '1-10'
        else:
            return '0'

def get_tags(content):
    if content == '\n' or len(content) == 0:
        return ['null']
    return content.strip().split(',')

def rewrite_train_test(input_file, rewrite_file):
    namecol = {}
    reader = open(input_file, 'r')
    writer = open(rewrite_file, 'w')
    first = True
    for line in reader:
        s = line.strip().split('\t')
        if first:
            first = False
            for i in range(1, len(s)):
                name = s[i]
                if name in feature_cols or name in feature_cols_special:
                    namecol[name] = i
            continue
        writer.write('%s,%.5f' % (s[0], w_0))
        fid = 1
        for name in feature_cols + feature_cols_special:
            i = namecol[name]
            feature = s[i]
            if name in feature_cols_special:
                feature = feat_trans(name, s[i])
            vfeat = np.zeros(k)
            wfeat = 0.
            if ',' in feature:
                tags = get_tags(feature)
                for tag in tags:
                    feature = name + ':' + tag
                    if feature not in feature_index:
                        feature = name + ':other'
                    idx = feature_index[feature]
                    vfeat += v[idx]
                    wfeat += w[idx]
                vfeat /= len(tags)
                wfeat /= len(tags)
            else:
                feature = name + ':' + feature
                if feature not in feature_index:
                    feature = name + ':other'
                idx = feature_index[feature]
                vfeat = v[idx]
                wfeat = w[idx]
            writer.write(',%.5f' % wfeat)
            #writer.write(' %d:%.5f' % (fid, wfeat))
            fid += 1
            for j in range(k):
                writer.write(',%.5f' % vfeat[j])
                #writer.write(' %d:%.5f' % (fid, vfeat[j]))
                fid += 1
        writer.write('\n')
    reader.close()
    writer.close()

def pred_best(x,best_w,best_v,best_w_0):
    p = best_w_0
    sum_1 = 0
    sum_2 = 0
    for (feat, val) in x:
        tmp = best_v[feat] * val
        sum_1 += tmp
        sum_2 += tmp * tmp
    p = np.sum(sum_1 * sum_1 - sum_2) / 2.0 + w_0
    for (feat, val) in x:
        p += best_w[feat] * val
    p = sigmoid(p)
    return p
def pred_all(file,best_w,best_v,best_w_0):
    fi = open(file, 'r')
    yp = []
    for line in fi:
        data = one_data_y_x(line)
        pclk = pred_best(data[1],best_w,best_v,best_w_0)
        yp.append(pclk)
    fi.close()
    print type(yp)
    print len(yp)
    return yp 
    
# start here
# global setting
np.random.seed(10)
one_value = True
k = 10
learning_rate = 0.01
weight_decay = 1E-6
v_weight_decay = 1E-6
train_rounds = 40
buffer_num = 1000000

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
v = (np.random.rand(feature_num, k) - 0.5) * init_weight
w = np.zeros(feature_num)
w_0 = 0

# train
best_auc = 0.
best_w=w
best_v=v
best_w_0=w_0
overfitting = False
print 'training:'
fo = open(result_file, 'a')
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
                (p, vsum) = pred(x)
                d = y - p
                w_0 = w_0 * (1 - weight_decay) + learning_rate * d
                for (feat, val) in x:
                    w[feat] = w[feat] * (1 - weight_decay) + learning_rate * d * val
                for (feat, val) in x:
                    v[feat] = v[feat] * (1 - v_weight_decay) + learning_rate * d * (val * vsum - v[feat] * val * val)
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
        pclk = pred(data[1])[0]
        y.append(clk)
        yp.append(pclk)
    fi.close()
    auc = roc_auc_score(y, yp)
    rmse = math.sqrt(mean_squared_error(y, yp))
    print '%d\t%.8f\t%.8f\t%dm%ds' % (round, auc, rmse, train_min, train_sec)
    fo.write('%d\t%.8f\t%.8f\t%dm%ds\n' % (round, auc, rmse, train_min, train_sec))
    fo.flush()
    if overfitting and auc < best_auc:
        print 'rewriting ' + trainlog_file + ' into ' + traindl_file
        rewrite_train_test(trainlog_file, traindl_file)
        print 'rewriting ' + testlog_file + ' into ' + testdl_file
        rewrite_train_test(testlog_file, testdl_file)
        print 'output model into ' + model_file
        output_model(model_file)
        break # stop training when overfitting two rounds already
    if auc > best_auc:
        best_auc = auc
        best_w=w
        best_v=v
        best_w_0=w_0
        overfitting = False
    else:
        overfitting = True
        
ut.save_weights("fm_train_"+advertiser+".p",pred_all(train_file,best_w,best_v,best_w_0))
ut.save_weights("fm_test_"+advertiser+".p",pred_all(test_file,best_w,best_v,best_w_0))
fo.close()
