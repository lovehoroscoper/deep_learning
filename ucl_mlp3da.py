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
import ucl_denosing_autoencoder as da
import data_fm as fm
import pickle
import dl_utils as dlut

rng = numpy.random
rng.seed(1234)
batch_size=100                                                       #batch size
lr=0.01                                                                #learning rate
lambda1=0.001                                                        #regularisation rate
hidden1 = 300 #hidden layer 1
hidden2 = 100 #hidden layer 2
acti_type='tanh'                                                    #activation type
epoch = 100                                                               #epochs number
advertiser = '2997'
if len(sys.argv) > 1:
    advertiser = sys.argv[1]
train_file='../../make-ipinyou-data/' + advertiser + '/train.fm.txt'             #training file
test_file='../../make-ipinyou-data/' + advertiser + '/test.fm.txt'                   #test file
fm_model_file='../../make-ipinyou-data/' + advertiser + '/fm.model.txt'                   #fm model file
#feats = ut.feats_len(train_file)              

# get all lines into mem
farr=[]
yarr=[]
train_size=0
fi = open(train_file, 'r')
for line in fi:
    if line.strip() != '':
		s = line.strip().replace(':', ' ').split(' ')
		fi=[]
		for f in range(1, len(s), 2):
			if int(s[f+1])==1:
				fi.append(int(s[f]))
		farr.append(fi)
		yarr.append(int(s[0]))
		train_size+=1
farr = numpy.array(farr, dtype = numpy.int32)
yarr = numpy.array(yarr, dtype = numpy.int32)


test_size=ut.file_len(test_file)                      #test size
n_batch=train_size/batch_size                                        #number of batches
x_dim=133465

if advertiser == '2997':
    lr=0.05
if advertiser== '3386':                                       #number of batches
    x_dim=0
    
if sys.argv[2]=='mod' and advertiser=='2997':
    lr=0.1
    lambda1=0.00
    


    
    
ut.log_p('X:'+str(x_dim) + ' | Hidden 1:'+str(hidden1)+ ' | Hidden 2:'+str(hidden2)+
        ' | L rate:'+str(lr)+ ' | activation1:'+ str(acti_type)+
        ' | lambda:'+str(lambda1)
        )
        

ww3=numpy.zeros(hidden2)
# ww3=rng.uniform(-0.05,0.05,hidden2)
bb3=0.


arr=[]
arr.append(x_dim)
arr.append(hidden1)
arr.append(hidden2)

# ww1,bb1,ww2,bb2=da.get_da_weights(train_file,arr,ncases=train_size,batch_size=100000)
# pickle.dump( (ww1,bb1,ww2,bb2), open( "2997_da_10.p", "wb" ))

# (ww1,bb1,ww2,bb2)=pickle.load( open( "2997_da_10.p", "rb" ) )
ww1,bb1=ut.init_weight(x_dim,hidden1,'sigmoid')
ww2,bb2=ut.init_weight(hidden1,hidden2,'sigmoid')

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
# z1=T.dot(x, w1) + b1
# if acti_type=='sigmoid':
#     h1 = 1 / (1 + T.exp(-z1))              # hidden layer 1
# elif acti_type=='linear':
#     h1 = z1
# elif acti_type=='tanh':
#     h1=T.tanh(z1)

z2=T.dot(x, w2) + b2
if acti_type=='sigmoid':
    h2 = 1 / (1 + T.exp(-z2))              # hidden layer 2
elif acti_type=='linear':
    h2 = z2
elif acti_type=='tanh':
    h2=T.tanh(z2)

p_1 = 1 / (1 + T.exp(-T.dot(h2, w3) - b3))               # Probability that target = 1
prediction = p_1 #> 0.5                                   # The prediction thresholded
xent = - y * T.log(p_1) - (1-y) * T.log(1-p_1)             # Cross-entropy loss function
cost = xent.mean() + lambda1 * (
       (w2 ** 2).sum() + (w3 ** 2).sum() + (b2 ** 2).sum() + (b3 ** 2))    # The cost to minimize
gw3, gb3, gw2, gb2, gx = T.grad(cost, [w3, b3, w2, b2, x])        # Compute the gradient of the cost


# Compile
train = theano.function(
          inputs=[x,y],
          outputs=[gx, w1,w2, w3,b1,b2,b3],updates=(
          (w2, w2 - lr * gw2), (b2, b2 - lr * gb2),
          (w3, w3 - lr * gw3), (b3, b3 - lr * gb3)))
predict = theano.function(inputs=[x], outputs=prediction)


#print error
def print_err(file,msg=''):
    auc,rmse=whole_auc_rmse(file)
    ut.log_p( msg + '\t' + str(auc) + '\t' + str(rmse))    



#get error via batch
def auc_rmse(file,err_batch=1000):
    yp = []
    fi = open(file, 'r')
    flag_start=0
    flag=False
    xarray = []
    yarray=[]
    start_t = time.clock()
    while True:
        line=fi.readline()
        if len(line.strip()) == 0:
            flag=True
        else:
            flag_start+=1
            if flag==False:
                x_dense=numpy.zeros(ww1.shape[1])
                s = line.strip().replace(':', ' ').split(' ')
                for f in range(1, len(s), 2):
                    if int(s[f+1])==1:
                        x_dense += ww1[f]
                x_dense+=bb1
                x_dense=(1.0 / (1.0 + numpy.exp(-x_dense)))
                xarray.append(x_dense)
                yarray.append(int(s[0]))
        if ((flag_start==err_batch) or (flag==True)):
#             print 'one epoch',time.clock()-start_t
            pred=predict(xarray)
            for p in pred:
                yp.append(p)
            flag_start=0
            xarray=[]
        if flag==True:
            break
    fi.close()
    auc = roc_auc_score(yarray, yp)
    rmse = math.sqrt(mean_squared_error(yarray, yp))
    return auc,rmse

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
            x_dense=numpy.zeros(ww1.shape[1])
            s = line.strip().replace(':', ' ').split(' ')
            for f in range(1, len(s), 2):
                if int(s[f+1])==1:
                    x_dense += ww1[int(s[f])]
            x_dense+=bb1
            x_dense=(1.0 / (1.0 + numpy.exp(-x_dense)))
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
    xarray = []
    for i in range(index, index + size):
		x_dense=numpy.zeros(ww1.shape[1])
		fi=farr[i]
		for f in fi:
			x_dense += ww1[f]
		x_dense+=bb1
		x_dense=(1.0 / (1.0 + numpy.exp(-x_dense)))
		xarray.append(x_dense)
    xarray = numpy.array(xarray, dtype = theano.config.floatX)
    return farr[index:(index+size),:],xarray,yarr[index:(index+size)]

# print_err(train_file,'\t\tTraining Err: \t' )# train error
# print_err(test_file,'\t\tTest Err: \t' )
# Train
print "Training model:"
min_err = 0
min_err_epoch = 0
times_reduce = 0
for i in range(epoch):
    start_time = time.time()
    index = 1
    for j in range(n_batch):
        start_t = time.clock()
        fi,h1,y = get_fi_h1_y(train_file,index,batch_size)
        index += len(fi)
        gx, ww1,ww2, ww3,bb1,bb2,bb3 = train(h1,y)
        b_size = len(fi)
        for t in range(b_size):
            ft = fi[t]
            gxt = gx[t]
            xt=h1[t]
            for feat in ft:
                ww1[feat]=ww1[feat]-lr * gxt*xt*(1-xt)
                #ww1[feat]=ww1[feat]* (1 - 2. * lambda1 * lr / b_size)-lr * (gxt*(1-gxt))

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
    if times_reduce<-2:
        break
ut.log_p( 'Minimal test error is '+ str( min_err)+' , at EPOCH ' + str(min_err_epoch))
