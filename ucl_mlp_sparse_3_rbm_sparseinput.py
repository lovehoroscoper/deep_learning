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
import ucl_gaussian_binary_rbm_sparse as gbrbm
import data_fm as fm
import pickle

rng = numpy.random
rng.seed(1234)
batch_size=1000                                                       #batch size
lr=0.01                                                                #learning rate
lambda1=0.0005                                                        #regularisation rate
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
#feats = ut.feats_len(train_file)                                           #feature size
train_size=312437        #ut.file_len(train_file)                    #training size
test_size=156063         #ut.file_len(test_file)                      #test size
n_batch=train_size/batch_size                                        #number of batches



o_fm=fm.DataFM(fm_model_file)



ut.log_p('X:'+str(o_fm.xdim) + ' | Hidden 1:'+str(hidden1)+ ' | Hidden 2:'+str(hidden2)+
        ' | L rate:'+str(lr)+ ' | activation1:'+ str(acti_type)+
        ' | lambda:'+str(lambda1)
        )
        

ww3=numpy.zeros(hidden2)

bb3=0.


arr=[]
# print o_fm.xdim
x_dim=133465
arr.append(x_dim)
arr.append(hidden1)
arr.append(hidden2)

# ww1,bb1,ww2,bb2=gbrbm.get_rbm_weights(train_file,arr,ncases=train_size,batch_size=100000,fm_model_file=fm_model_file)
# pickle.dump( (ww1,bb1,ww2,bb2), open( "sparse2997_1000.p", "wb" ))

(ww1,bb1,ww2,bb2)=pickle.load( open( "sparse2997_1000.p", "rb" ) )


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
          outputs=[gx, w2, w3],updates=(
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
# def getvector(v):
# # 	print 'v',v
# 	arr=[]
# 	index=0
# 	for i in range(x_dim):
# 		bool=False
# 		for l in v:
# 			a,b=l.split(":")
# 			if i==int(a):
# 				bool=True
# 		if bool:
# 			arr.append(int(b))
# 			index+=1
# 		else:
# 			arr.append(0)
# 	return arr
def get_fake_line(line):
	newline=[]
	for l in line:
		a,b=l.split(":")
		newline.append(l)
		newline.append(str(int(a)-1)+":0")
	return newline
def get_fi_h1_y(file,index,w1,b1,size):
	farray=[]
	xarray = []
	yarray=[]
	for i in range(index, index + size):
		line = linecache.getline(file, i)
		if line.strip() != '':
			x_dense=numpy.zeros(w1.shape[1])
			s = line.strip().replace(':', ' ').split(' ')
			y = int(s[0])
			fi=[]
			for f in range(1, len(s), 2):
				if int(s[f+1])==1:
					fi.append(int(s[f]))
			for i in range(w1.shape[1]):
				sum=0
				for j in range(1, len(s), 2):
					if int(s[f+1])==1:
						sum+=w1[int(s[f])][i]
				sum+=b1[i]
				x_dense[i]=(1 / (1 + math.exp(-sum)))
			farray.append(fi)
			xarray.append(x_dense)
			yarray.append(int(y))
	farray = numpy.array(farray, dtype = numpy.int32)
	xarray = numpy.array(xarray, dtype = theano.config.floatX)
	yarray = numpy.array(yarray, dtype = numpy.int32)
	return farray,xarray,yarray

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
        fi,h1,y = get_fi_h1_y(train_file,index,ww1,bb1,batch_size)
#         print 'fi',fi
#         print time.clock()-start_t
        index += batch_size
        gx, w2, w3 = train(h1,y)
        b_size = len(fi)
        for t in range(b_size):
            ft = fi[t]
            gxt = gx[t]
            for feat in ft:
            	for l in range(ww1.shape[1]):
            		ww1[feat]=ww1[feat]-lr * (gxt*(1-gxt))
        print 'one batch 1000 time',time.clock()-start_t
                # for l in range(o_fm.k):
#                     o_fm.feat_weights[feat][l] = o_fm.feat_weights[feat][l] * (1 - 2. * lambda1 * lr / b_size) \
#                                             - lr * gxt[o_fm.feat_layer_one_index(feat, l)] * 1
    

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
