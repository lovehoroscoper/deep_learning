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
lr=0.01                                                                #learning rate
lambda1=0.01                                                        #regularisation rate
hidden1 = 100                                                           #hidden layer 1
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



ut.log_p('Hidden one:'+str(hidden1)+'|L rate:'+str(lr)+
        '|activation1:'+ str(acti_type)+'|feats:'+str(xdim)+
        '|lambda1:'+str(lambda1)
        )
        
# initialise parameters
w=rng.uniform(    low=-numpy.sqrt(6. / (xdim + hidden1)),
                high=numpy.sqrt(6. / (xdim + hidden1)),
                size=(xdim,hidden1))
if acti_type=='sigmoid':
    ww1=numpy.asarray((w))
elif acti_type=='tanh':
    ww1=numpy.asarray((w*4))
else:
    ww1=numpy.asarray(rng.uniform(-1,1,size=(xdim,hidden1)))
    
ww2=numpy.zeros(hidden1)
bb1=numpy.zeros(hidden1)


# Declare Theano symbolic variables
x = T.matrix("x")
y = T.vector("y")
w1 = theano.shared(ww1, name="w1")
w2 = theano.shared(ww2, name="w2")
b1 = theano.shared(bb1, name="b1")
b2 = theano.shared(0., name="b2")


# Construct Theano expression graph
z=T.dot(x, w1) + b1
if acti_type=='sigmoid':
    h1 = 1 / (1 + T.exp(-z))              # hidden layer 1
elif acti_type=='linear':
    h1 = z
elif acti_type=='relu':
    h1=T.maximum(T.cast(0., theano.config.floatX), x)
    h1 = (z)*(z>0)
elif acti_type=='tanh':
    h1=T.tanh(z)
p_1 = 1 / (1 + T.exp(-T.dot(h1, w2) - b2))               # Probability that target = 1
prediction = p_1 #> 0.5                                   # The prediction thresholded
xent = - y * T.log(p_1) - (1-y) * T.log(1-p_1)             # Cross-entropy loss function
cost = xent.sum() + lambda1 * ((w1 ** 2).sum() +
       (w2 ** 2).sum() + (b1 ** 2).sum() + (b2 ** 2))    # The cost to minimize
gw2, gb2, gw1, gb1, gx= T.grad(cost, [w2, b2, w1, b1, x])        # Compute the gradient of the cost
                                     
#outputgx = theano.function(inputs=[x,y], outputs=gx[0][0])
outputgx = theano.function(inputs=[x,y], outputs=gx)

# Compile
train = theano.function(
          inputs=[x,y],
          outputs=[gx,w1,w2],updates=(
          (w1, w1 - lr * gw1), (b1, b1- lr * gb1),
          (w2, w2 - lr * gw2), (b2, b2 - lr * gb2)))
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
    xarray=[]
    yarray=[]
    feat_val = {}
    for i in range(index,index+size):
        line=linecache.getline(file, i)
        if line.strip() != '':
            f, x, y = get_fxy(line.strip())
            xarray.append(x)
            yarray.append(y)
            for feat in f:
                if feat not in feat_val:
                    feat_val[feat] = 0.
                feat_val[feat] += 1.
    xarray=numpy.array(xarray, dtype=theano.config.floatX)
    yarray=numpy.array(yarray, dtype=numpy.int32)
    for feat in feat_val:
        feat_val[feat] /= size
    return feat_val, xarray,yarray

def get_xy(line):
    s = line.replace(':', ' ').split()
    y=int(s[0])
    feats = [int(s[j]) for j in range(1, len(s), 2)]
    x = feats_to_layer_one_array(feats)
    return x,y

def get_fxy(line):
    s = line.replace(':', ' ').split()
    y=int(s[0])
    feats = [int(s[j]) for j in range(1, len(s), 2)]
    x = feats_to_layer_one_array(feats)
    return feats,x,y

#print_err(test_file,'InitTestErr:')


# Train
print "Training model:"
min_err=0
min_err_epoch=0
times_reduce=0
for i in range(epoch):
    start_time = time.time()
    index=1
    for j in range(n_batch):
        f,x,y=get_batch_data(train_file,index,batch_size)
        index+=batch_size
        gx, w1, w2 = train(x,y)

        # if i == 1:
        #     print 'gx:'
        #     print str(numpy.asarray(gx))
        #     print '\nw1:'
        #     print str(numpy.asarray(w1))
        #     print '\nw2:'
        #     print str(numpy.asarray(w2))
        #     exit(234)

        #print 'gx[0][0]\t%.5f' % outputgx(numpy.asarray(x[0]).reshape(1, feats), numpy.asarray(y[0]).reshape(1))
        #gx = numpy.average(outputgx(numpy.asarray(x), numpy.asarray(y)), axis=0)
        gx = numpy.asarray(gx).sum(axis = 0)
        #gx = numpy.sum(gx, axis=0)
        #print gx.shape #str(len(gx)) + '\t' + str(len(gx[0]))
        #print str(gx)
        #print str(gx[0])
        #exit(123)

        # now update the feature layer weights
        #gx = numpy.average(outputgx(numpy.asarray(x), numpy.asarray(y)), axis=0)

        for feat in f:
            val = f[feat]
            for l in range(k):
                feat_weights[feat][l] = feat_weights[feat][l] \
                                        - lr * gx[feat_layer_one_index(feat, l)] * val ##* (1 - lambda1 * val)


    train_time = time.time() - start_time
    mins = int(train_time / 60)
    secs = int(train_time % 60)
    print 'mins:' + str(mins) + ',secs:' + str(secs)
    print_err(train_file,'\t\tTraining Err: \t' + str(i))# train error
    auc, rmse = get_err_bat(test_file)
    ut.log_p( 'Test Err:' + str(i) + '\t' + str(auc) + '\t' + str(rmse))
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
