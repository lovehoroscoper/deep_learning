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

rng = numpy.random
rng.seed(1234)
batch_size=100                                                          #batch size
lr=0.002                                                                #learning rate
lambda1=0.1 # .01                                                        #regularisation rate
hidden1 = 300 #hidden layer 1
hidden2 = 100 #hidden layer 2
hidden3 = 100
hidden4 = 100
acti_type='tanh'                                                    #activation type
epoch = 100                                                               #epochs number
advertiser = '2997'
if len(sys.argv) > 1:
    advertiser = sys.argv[1]
train_file='../../make-ipinyou-data/' + advertiser + '/train.fm.txt'             #training file
test_file='../../make-ipinyou-data/' + advertiser + '/test.fm.txt'                   #test file
fm_model_file='../../make-ipinyou-data/' + advertiser + '/fm.model.txt'                   #fm model file
#feats = ut.feats_len(train_file)                                           #feature size
train_size=312437        #ut.file_len(train_file)                    #training size
test_size=156063         #ut.file_len(test_file)                      #test size
n_batch=train_size/batch_size                                        #number of batches



name_field = {'weekday':0, 'hour':1, 'useragent':2, 'IP':3, 'region':4, 'city':5, 'adexchange':6, 'domain':7, 'slotid':8,
       'slotwidth':9, 'slotheight':10, 'slotvisibility':11, 'slotformat':12, 'creative':13, 'advertiser':14, 'slotprice':15}

feat_field = {}
feat_weights = {}
w_0 = 0
feat_num = 0
k = 0
xdim = 0
fi = open(fm_model_file, 'r')
first = True
for line in fi:
    s = line.strip().split()
    if first:
        first = False
        w_0 = float(s[0])
        feat_num = int(s[1])
        k = int(s[2]) + 1 # w and v
        xdim = 1 + len(name_field) * k
    else:
        feat = int(s[0])
        weights = [float(s[1 + i]) for i in range(k)]
        feat_weights[feat] = weights
        name = s[1 + k][0:s[1 + k].index(':')]
        field = name_field[name]
        feat_field[feat] = field

def feat_layer_one_index(feat, l):
    return 1 + feat_field[feat] * k + l

def feats_to_layer_one_array(feats):
    x = numpy.zeros(xdim)
    x[0] = w_0
    for feat in feats:
        x[feat_layer_one_index(feat, 0):feat_layer_one_index(feat, k)] = feat_weights[feat]
    return x

ut.log_p('X:'+str(xdim) + ' | Hidden 1:'+str(hidden1)+ ' | Hidden 2:'+str(hidden2)+
        ' | L rate:'+str(lr)+ ' | activation1:'+ str(acti_type)+
        ' | lambda:'+str(lambda1)
        )
        
# initialise parameters
ww1,bb1=ut.init_weight(xdim,hidden1,acti_type)
ww2,bb2=ut.init_weight(hidden1,hidden2,acti_type)
ww3,bb3=ut.init_weight(hidden2,hidden3,acti_type)
ww4,bb4=ut.init_weight(hidden3,hidden4,acti_type)
ww5=numpy.zeros(hidden2)
bb5=0.

# Declare Theano symbolic variables
x = T.matrix("x")
y = T.vector("y")
w1 = theano.shared(ww1, name="w1")
w2 = theano.shared(ww2, name="w2")
w3 = theano.shared(ww3, name="w3")
w4 = theano.shared(ww4, name="w4")
w5 = theano.shared(ww5, name="w5")
b1 = theano.shared(bb1, name="b1")
b2 = theano.shared(bb2, name="b2")
b3 = theano.shared(bb3 , name="b3")
b4 = theano.shared(bb4 , name="b4")
b5 = theano.shared(bb5 , name="b5")

# Construct Theano expression graph
z1=T.dot(x, w1) + b1
if acti_type=='sigmoid':
    z1 = 1 / (1 + T.exp(-z1))              # hidden layer 1
elif acti_type=='linear':
    z1 = z1
elif acti_type=='tanh':
    z1=T.tanh(z1)

z2=T.dot(z1, w2) + b2
if acti_type=='sigmoid':
    z2 = 1 / (1 + T.exp(-z2))              # hidden layer 2
elif acti_type=='linear':
    z2 = z2
elif acti_type=='tanh':
    z2=T.tanh(z2)
    
z3=T.dot(z2, w3) + b3
z3=T.tanh(z3)

z4=T.dot(z3, w4) + b4
z4=T.tanh(z4)

p_1 = 1 / (1 + T.exp(-T.dot(z4, w5) - b5))               # Probability that target = 1
prediction = p_1 #> 0.5                                   # The prediction thresholded
xent = - y * T.log(p_1) - (1-y) * T.log(1-p_1)             # Cross-entropy loss function
cost = xent.sum() + lambda1 * ((w1 ** 2).sum() +
       (w2 ** 2).sum() + (w3 ** 2).sum() +
       (w4 ** 2).sum() + (w5 ** 2).sum() +
       (b1 ** 2).sum() + (b2 ** 2).sum() + 
       (b3 ** 2).sum() + (b4 ** 2).sum() +
       (b5 ** 2))    # The cost to minimize
gw5, gb5,gw4, gb4,gw3, gb3, gw2, gb2, gw1, gb1, gx = T.grad(cost, [w5, b5,w4, b4,w3, b3, w2, b2, w1, b1, x])        # Compute the gradient of the cost


# Compile
train = theano.function(
          inputs=[x,y],
          outputs=[gx, w1, w2, w3],updates=(
          (w1, w1 - lr * gw1), (b1, b1 - lr * gb1),
          (w2, w2 - lr * gw2), (b2, b2 - lr * gb2),
          (w3, w3 - lr * gw3), (b3, b3 - lr * gb3),
          (w4, w4 - lr * gw4), (b4, b4 - lr * gb4),
          (w5, w5 - lr * gw5), (b5, b5 - lr * gb5)
          ))
predict = theano.function(inputs=[x], outputs=prediction)


#print error
def print_err(file,msg=''):
    auc,rmse=get_err_bat(file)
    ut.log_p( msg + '\t' + str(auc) + '\t' + str(rmse))    


#get error via batch
def get_err_bat(file,err_batch=100000):
    y = []
    yp = []
    fi = open(file, 'r')
    flag_start=0
    xx_bat=[]
    flag=False
    while True:
        line=fi.readline()
        if len(line) == 0:
            flag=True
        flag_start+=1
        if flag==False:
            xx,yy = get_xy(line)
            xx_bat.append(numpy.asarray(xx))
        if ((flag_start==err_batch) or (flag==True)):
            pred=predict(xx_bat)
            for p in pred:
                yp.append(p)
            flag_start=0
            xx_bat=[]
        if flag==False:
            y.append(yy)
        if flag==True:
            break
    fi.close()
    auc = roc_auc_score(y, yp)
    rmse = math.sqrt(mean_squared_error(y, yp))
    return auc,rmse

def get_batch_data(file,index,size):#1,5->1,2,3,4,5
    xarray = []
    yarray = []
    farray = []
    for i in range(index, index + size):
        line = linecache.getline(file, i)
        if line.strip() != '':
            f, x, y = get_fxy(line.strip())
            xarray.append(x)
            yarray.append(y)
            farray.append(f)
    xarray = numpy.array(xarray, dtype = theano.config.floatX)
    yarray = numpy.array(yarray, dtype = numpy.int32)
    return farray, xarray, yarray

def get_xy(line):
    s = line.replace(':', ' ').split()
    y = int(s[0])
    feats = [int(s[j]) for j in range(1, len(s), 2)]
    x = feats_to_layer_one_array(feats)
    return x, y

def get_fxy(line):
    s = line.replace(':', ' ').split()
    y=int(s[0])
    feats = [int(s[j]) for j in range(1, len(s), 2)]
    x = feats_to_layer_one_array(feats)
    return feats,x,y

#print_err(test_file,'InitTestErr:')


# Train
print "Training model:"
min_err = 0
min_err_epoch = 0
times_reduce = 0
for i in range(epoch):
    start_time = time.time()
    index = 1
    for j in range(n_batch):
        f,x,y = get_batch_data(train_file,index,batch_size)
        index += batch_size
        gx, w1, w2, w3 = train(x,y)

        b_size = len(f)
        for t in range(b_size):
            ft = f[t]
            gxt = gx[t]
            for feat in ft:
                for l in range(k):
                    feat_weights[feat][l] = feat_weights[feat][l] * (1 - 2. * lambda1 * lr / b_size) \
                                            - lr * gxt[feat_layer_one_index(feat, l)] * 1




    train_time = time.time() - start_time
    mins = int(train_time / 60)
    secs = int(train_time % 60)
    print 'training: ' + str(mins) + 'm ' + str(secs) + 's'

    start_time = time.time()
    print_err(train_file,'\t\tTraining Err: \t' + str(i))# train error
    train_time = time.time() - start_time
    mins = int(train_time / 60)
    secs = int(train_time % 60)
    print 'training error: ' + str(mins) + 'm ' + str(secs) + 's'

    start_time = time.time()
    auc, rmse = get_err_bat(test_file)
    test_time = time.time() - start_time
    mins = int(test_time / 60)
    secs = int(test_time % 60)
    ut.log_p( 'Test Err:' + str(i) + '\t' + str(auc) + '\t' + str(rmse))
    print 'test error: ' + str(mins) + 'm ' + str(secs) + 's'

    #stop training when no improvement for a while 
    if auc>min_err:
        min_err=auc
        min_err_epoch=i
        if times_reduce<3:
            times_reduce+=1
    else:
        times_reduce-=1
    if times_reduce<-3:
        break
ut.log_p( 'Minimal test error is '+ str( min_err)+' , at EPOCH ' + str(min_err_epoch))
