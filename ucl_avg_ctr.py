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
import data_fm as fm
import pickle
import ucl_denosing_autoencoder as da
rng = numpy.random
rng.seed(1234)


batch_size=10000                                                          #batch size
lr=0.1                                                               #learning rate
lambda1=0.01# .01 
#hidden0=300                                                       #regularisation rate
hidden1 = 300 #hidden layer 1
hidden2 = 100 #hidden layer 2
acti_type='tanh'                                                    #activation type
epoch = 100                                                               #epochs number
advertiser = '2997'
if len(sys.argv) > 1:
    advertiser = sys.argv[1]
train_file='../../make-ipinyou-data/' + advertiser + '/train.fm.txt'             #training file
test_file='../../make-ipinyou-data/' + advertiser + '/test.fm.txt'                   #test file
index_file='../../make-ipinyou-data/' + advertiser + '/featindex.fm.txt'

train_size=ut.file_len(train_file)                    #training size
test_size=ut.file_len(test_file)                      #test size
n_batch=train_size/batch_size                                        #number of batches


print 'reading feature index'
feature_index = {}
index_feature = {}
max_feature_index = 0
feature_num = 0
fi = open(index_file, 'r')
feat_name=""
feat_num=0
for line in fi:
    s = line.strip().split('\t')
    index = int(s[1])
    if feat_name!=s[0].split(':')[0]:
        feat_name=s[0].split(':')[0]
        feat_num+=1
    feature_index[s[0]] = index
    index_feature[index] = s[0]
    max_feature_index = max(max_feature_index, index)
fi.close()
feature_num = max_feature_index + 1
x_dim=feature_num
hidden0=feat_num
# print 'x_dim:',x_dim
# print 'hidden0: ' ,hidden0
clicks=numpy.zeros(feature_num)
impressions=numpy.zeros(feature_num)
fi = open(train_file, 'r')
flag_start=0
while True:
    line=fi.readline()
    if len(line) == 0:
        break
    else:
        s = line.replace(':', ' ').split()
        y = int(s[0])
        for j in range(1, len(s), 2):
            impressions[int(s[j])]+=1
            if y==1:
                clicks[int(s[j])]+=1
clicks=clicks/impressions
clicks[numpy.isnan(clicks)] = 0
# print clicks
fi.close()


ut.log_p('bat size:'+str(batch_size)+'|X:'+str(x_dim) + ' | Hidden 0:'+str(hidden0)+ ' | Hidden 1:'+str(hidden1)+ ' | Hidden 2:'+str(hidden2)+
        ' | L rate:'+str(lr)+ ' | activation1:'+ str(acti_type)+
        ' | lambda:'+str(lambda1)
        )
        
# initialise parameters
arr=[]
arr.append(x_dim)
arr.append(hidden0)
arr.append(hidden1)
arr.append(hidden2)

# ww0,bb0=ut.init_weight(x_dim,hidden0,'sigmoid')
ww0=numpy.asarray(clicks)
bb0=numpy.zeros(hidden0)
ww1,bb1=ut.init_weight(hidden0,hidden1,'sigmoid')
ww2,bb2=ut.init_weight(hidden1,hidden2,'sigmoid')
# ww0,bb0,ww1,bb1,ww2,bb2=da.get_da_weights(train_file,arr,ncases=train_size,batch_size=100000)
# pickle.dump( (ww0,bb0,ww1,bb1,ww2,bb2), open( "2997_da_4l_10.p", "wb" ))

# (ww0,bb0,ww1,bb1,ww2,bb2)=pickle.load( open( "2997_da_4l_10.p", "rb" ) )


# ww2,bb2=ut.init_weight(hidden1,hidden2,'sigmoid')
ww3=rng.uniform(-0.05,0.05,hidden2)
ww3=numpy.zeros(hidden2)
#
bb3=0.


arr=[]
arr.append(x_dim)
arr.append(hidden1)
arr.append(hidden2)


ww3=numpy.reshape(ww3,hidden2)
bb3=float(bb3)


# Declare Theano symbolic variables
x = T.matrix("x")
y = T.vector("y")
w1 = theano.shared(ww1, name="w1", borrow=True)
w2 = theano.shared(ww2, name="w2", borrow=True)
w3 = theano.shared(ww3, name="w3", borrow=True)
b1 = theano.shared(bb1, name="b1", borrow=True)
b2 = theano.shared(bb2, name="b2", borrow=True)
b3 = theano.shared(bb3 , name="b3", borrow=True)


# Construct Theano expression graph
z1=T.dot(x, w1) + b1
if acti_type=='sigmoid':
    h1 = 1 / (1 + T.exp(-z1))              # hidden layer 1
elif acti_type=='linear':
    h1 = z1
elif acti_type=='tanh':
    h1=T.tanh(z1)

z2=T.dot(z1, w2) + b2
if acti_type=='sigmoid':
    h2 = 1 / (1 + T.exp(-z2))              # hidden layer 2
elif acti_type=='linear':
    h2 = z2
elif acti_type=='tanh':
    h2=T.tanh(z2)

p_1 = 1 / (1 + T.exp(-T.dot(h2, w3) - b3))               # Probability that target = 1
prediction = p_1 #> 0.5                                   # The prediction thresholded
xent = - y * T.log(p_1) - (1-y) * T.log(1-p_1)             # Cross-entropy loss function
cost = xent.mean() + lambda1 * ((w1 ** 2).sum() +
       (w2 ** 2).sum() + (w3 ** 2).sum() +
       (b1 ** 2).sum() + (b2 ** 2).sum() + (b3 ** 2))    # The cost to minimize
gw3, gb3, gw2, gb2, gw1, gb1, gx = T.grad(cost, [w3, b3, w2, b2, w1, b1, x])        # Compute the gradient of the cost


# Compile
train = theano.function(
          inputs=[x,y],
          outputs=[gx, w1,w2, w3,b1,b2,b3],updates=(
          (w1, w1 - lr * gw1), (b1, b1 - lr * gb1),
          (w2, w2 - lr * gw2), (b2, b2 - lr * gb2),
          (w3, w3 - lr * gw3), (b3, b3 - lr * gb3)))
predict = theano.function(inputs=[x], outputs=prediction)


#print error
def print_err(file,msg=''):
    auc,rmse=whole_auc_rmse(file)
    ut.log_p( msg + '\t' + str(auc) + '\t' + str(rmse))    


#get error via batch
def whole_auc_rmse(file):
    yp = []
    fi = open(file, 'r')
    xarray = []
    yarray=[]
    while True:
        line=fi.readline()
        if len(line.strip()) == 0:
            break
        else:
            x_dense=numpy.zeros(hidden0)
            s = line.strip().replace(':', ' ').split(' ')
            i=0
            for f in range(1, len(s), 2):
                if int(s[f+1])==1:
                    x_dense[i]= ww0[int(s[f])]
                    x_dense[i]+=bb0[i]
                    i+=1
            xarray.append(x_dense)
            yarray.append(int(s[0]))
    yp=predict(xarray)
    flag_start=0
    xarray=[]
    fi.close()
    auc = roc_auc_score(yarray, yp)
    rmse = math.sqrt(mean_squared_error(yarray, yp))
    return auc,rmse
def get_fi_h1_y(file,index,size):
    farray=[]
    xarray = []
    yarray=[]
    for i in range(index, index + size):
        line = linecache.getline(file, i)
        if line.strip() != '':
            x_dense=numpy.zeros(hidden0)
            s = line.strip().replace(':', ' ').split(' ')
#             print 's:',s
            fi=[]
            i=0
            for f in range(1, len(s), 2):
                if int(s[f+1])==1:
                    fi.append(int(s[f]))
                    x_dense[i]= ww0[int(s[f])]
                    x_dense[i]+=bb0[i]
                    i+=1
#             x_dense=(1.0 / (1.0 + numpy.exp(-x_dense)))
            farray.append(fi)
            xarray.append(x_dense)
            yarray.append(int(s[0]))
    farray = numpy.array(farray, dtype = numpy.int32)
    xarray = numpy.array(xarray, dtype = theano.config.floatX)
    yarray = numpy.array(yarray, dtype = numpy.int32)
    return farray,xarray,yarray

# print_err(test_file,'InitTestErr:')


# Train
print "Training model:"
min_err = 0
min_err_epoch = 0
times_reduce = 0
for i in range(epoch):
    start_time = time.time()
    index = 1
    for j in range(n_batch):
        fi,x,y = get_fi_h1_y(train_file,index,batch_size)
        index += batch_size
        gx,ww1,ww2, ww3,bb1,bb2,bb3 = train(x,y)
        b_size = len(fi)
        for t in range(b_size):
            ft = fi[t]
            gxt = gx[t]
            ii=0
            for feat in ft:
                ww0[feat]=ww0[feat]*(1 - 2 * lambda1 * lr / b_size)-lr * gxt[ii]*1
                ii+=1


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
    auc, rmse = whole_auc_rmse(test_file)
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
    if times_reduce<=0:
        break
ut.log_p( 'Minimal test error is '+ str( min_err)+' , at EPOCH ' + str(min_err_epoch))

