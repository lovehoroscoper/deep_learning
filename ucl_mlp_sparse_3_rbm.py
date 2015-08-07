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

rng = numpy.random
rng.seed(1234)
batch_size=100                                                          #batch size
lr=0.001                                                               #learning rate
lambda1=0.00001 # .01                                                        #regularisation rate
hidden1 = 400 #hidden layer 1
hidden2 = 100 #hidden layer 2
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



o_fm=fm.DataFM(fm_model_file)



ut.log_p('X:'+str(o_fm.xdim) + ' | Hidden 1:'+str(hidden1)+ ' | Hidden 2:'+str(hidden2)+
        ' | L rate:'+str(lr)+ ' | activation1:'+ str(acti_type)+
        ' | lambda:'+str(lambda1)
        )
        
# initialise parameters
w=rng.uniform(  low=-numpy.sqrt(6. / (o_fm.xdim + hidden1)),
                high=numpy.sqrt(6. / (o_fm.xdim + hidden1)),
                size=(o_fm.xdim,hidden1))
if acti_type=='sigmoid':
    ww1=numpy.asarray((w))
elif acti_type=='tanh':
    ww1=numpy.asarray((w*4))
else:
    ww1=numpy.asarray(rng.uniform(-1,1,size=(o_fm.xdim,hidden1)))
#
bb1=numpy.zeros(hidden1)
#
#
v=rng.uniform(  low=-numpy.sqrt(6. / (hidden1 + hidden2)),
                high=numpy.sqrt(6. / (hidden1 + hidden2)),
                size=(hidden1,hidden2))
if acti_type=='sigmoid':
    ww2=numpy.asarray((v))
elif acti_type=='tanh':
    ww2=numpy.asarray((v*4))
else:
    ww2=numpy.asarray(rng.uniform(-1,1,size=(hidden1,hidden2)))
#
bb2=numpy.zeros(hidden2)
#
ww3=numpy.zeros(hidden2)
#
bb3=0.


arr=[]
arr.append(o_fm.xdim)
arr.append(hidden1)
arr.append(hidden2)

# ww1,bb1,ww2,bb2=gbrbm.get_rbm_weights(train_file,arr,ncases=train_size,batch_size=100000,fm_model_file=fm_model_file)
# pickle.dump( (ww1,bb1,ww2,bb2), open( "2997_rbm_400_100_epochs10.p", "wb" ))

(ww1,bb1,ww2,bb2)=pickle.load(open( "2997_rbm_400_100_epochs10.p", "rb" ) )


ww3=numpy.reshape(ww3,hidden2)
bb3=float(bb3)


# Declare Theano symbolic variables
x = T.matrix("x")
y = T.vector("y")
w1 = theano.shared(ww1, name="w1")
w2 = theano.shared(ww2, name="w2")
w3 = theano.shared(ww3, name="w3")
b1 = theano.shared(bb1, name="b1")
b2 = theano.shared(bb2, name="b2")
b3 = theano.shared(bb3 , name="b3")


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
cost = xent.sum() + lambda1 * ((w1 ** 2).sum() +
       (w2 ** 2).sum() + (w3 ** 2).sum() +
       (b1 ** 2).sum() + (b2 ** 2).sum() + (b3 ** 2))    # The cost to minimize
gw3, gb3, gw2, gb2, gw1, gb1, gx = T.grad(cost, [w3, b3, w2, b2, w1, b1, x])        # Compute the gradient of the cost


# Compile
train = theano.function(
          inputs=[x,y],
          outputs=[gx, w1, w2, w3],updates=(
          (w1, w1 - lr * gw1), (b1, b1 - lr * gb1),
          (w2, w2 - lr * gw2), (b2, b2 - lr * gb2),
          (w3, w3 - lr * gw3), (b3, b3 - lr * gb3)))
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
            xx,yy = o_fm.get_xy_fm(line)
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
        f,x,y = o_fm.get_batch_data(train_file,index,batch_size)
        index += batch_size
        gx, w1, w2, w3 = train(x,y)

        b_size = len(f)
        for t in range(b_size):
            ft = f[t]
            gxt = gx[t]
            for feat in ft:
                for l in range(o_fm.k):
                    o_fm.feat_weights[feat][l] = o_fm.feat_weights[feat][l] * (1 - 2. * lambda1 * lr / b_size) \
                                            - lr * gxt[o_fm.feat_layer_one_index(feat, l)] * 1

        #gx = numpy.asarray(gx).sum(axis = 0)

        # if i == 1:
        #     print 'gx:'
        #     print str(numpy.asarray(gx))
        #     print '\nw1:'
        #     print str(numpy.asarray(w1))
        #     print '\nw2:'
        #     print str(numpy.asarray(w2))
        #     print '\nw3:'
        #     print str(numpy.asarray(w3))
        #     print '\n'
            # exit(234)

        # now update the feature layer weights
        # for feat in f:
        #     val = f[feat]
        #     for l in range(k):
        #         feat_weights[feat][l] = feat_weights[feat][l] * (1 - lambda1 * lr) \
        #                                 - lr * gx[feat_layer_one_index(feat, l)] * val


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
