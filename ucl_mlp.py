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
batch_size=100					  									#batch size
lr=0.1																#learning rate
lambda1=0.001														#regularisation rate
hidden1 = 100									   					#hidden layer 1
acti_type='tanh'													#activation type
epoch = 100										 				  	#epochs number
advertiser = '2997'
if len(sys.argv) > 1:
    advertiser = sys.argv[1]
train_file='../../make-ipinyou-data/' + advertiser + '/train.dl.txt' 			#training file
test_file='../../make-ipinyou-data/' + advertiser + '/test.dl.txt'	   			#test file
feats = ut.feats_len(train_file)				   						#feature size
train_size=312437		#ut.file_len(train_file)					#training size
test_size=156063		 #ut.file_len(test_file)		  			#test size
n_batch=train_size/batch_size										#number of batches


ut.log_p('Hidden one:'+str(hidden1)+'|L rate:'+str(lr)+
		'|activation1:'+ str(acti_type)+'|feats:'+str(feats)+
		'|lambda1:'+str(lambda1)
		)
		
# initialise parameters
w=rng.uniform(	low=-numpy.sqrt(6. / (feats + hidden1)),
				high=numpy.sqrt(6. / (feats + hidden1)),
				size=(feats,hidden1))			
if acti_type=='sigmoid':
	ww1=numpy.asarray((w))
elif acti_type=='tanh':
	ww1=numpy.asarray((w*4))
else:
	ww1=numpy.asarray(rng.uniform(-1,1,size=(feats,hidden1)))
	
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
	h1 = 1 / (1 + T.exp(-z))  			# hidden layer 1
elif acti_type=='linear':
	h1 = z
elif acti_type=='relu':
	h1=T.maximum(T.cast(0., theano.config.floatX), x)
	h1 = (z)*(z>0)
elif acti_type=='tanh':
	h1=T.tanh(z)
p_1 = 1 / (1 + T.exp(-T.dot(h1, w2) - b2))   			# Probability that target = 1
prediction = p_1 #> 0.5				   				# The prediction thresholded
xent = - y * T.log(p_1) - (1-y) * T.log(1-p_1) 			# Cross-entropy loss function
cost = xent.mean() + lambda1 * ((w1 ** 2).sum() + 
	   (w2 ** 2).sum() + (b1 ** 2).sum() + (b2 ** 2))	# The cost to minimize
gw2, gb2, gw1, gb1 = T.grad(cost, [w2, b2, w1, b1])		# Compute the gradient of the cost
									 

# Compile
train = theano.function(
		  inputs=[x,y],
		  outputs=[prediction, xent],updates=(
		  (w1, w1 - lr * gw1), (b1, b1- lr * gb1),
		  (w2, w2 - lr * gw2), (b2, b2 - lr * gb2)))
predict = theano.function(inputs=[x], outputs=prediction)


#print error
def print_err(file,msg=''):
	auc,rmse=get_err_bat(file)
	ut.log_p( msg + '\t' + str(auc) + '\t' + str(rmse))	


#get error via batch
def get_err_bat(file,err_batch=1000):
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
			xx,yy = ut.get_xy(line)
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

	
	
# #get error
# def get_err(file):
# 	y = []
# 	yp = []
# 	fi = open(file, 'r')
# 	for line in fi:
# 		xx,yy = ut.get_xy(line)
# 		xx=numpy.asarray(xx).reshape(1,feats)
# 		y.append(yy)
# 		yp.append(predict(numpy.asarray(xx)))
# 	fi.close()
# 	auc = roc_auc_score(y, yp)
# 	rmse = math.sqrt(mean_squared_error(y, yp))
# 	return auc,rmse
	
	
#first prediction
print_err(test_file,'InitTestErr:')


# Train
print "Training model:"
min_err=0
min_err_epoch=0
times_reduce=0
for i in range(epoch):
	start_time = time.time()
	index=1
	for j in range(n_batch):
		x,y=ut.get_batch_data(train_file,index,batch_size)
		index+=batch_size
		train(x,y)
	train_time = time.time() - start_time
	mins = int(train_time / 60)
	secs = int(train_time % 60)
	print 'mins:' + str(mins) + ',secs:' + str(secs)
	print_err(train_file,'\t\tTraining Err: \t' + str(i))# train error
	auc,rmse=get_err_bat(test_file)
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
