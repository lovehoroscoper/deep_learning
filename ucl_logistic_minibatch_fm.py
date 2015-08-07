import numpy
import theano
import theano.tensor as T
import linecache
rng = numpy.random
import math
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error

BATCHSIZE=10                        #batch size
TRAIN_SIZE=312437                   #trainning size
TEST_SIZE=156063                    #test size
FEATS = 69                          #feature size
EPOCHs = 100                        #epochs number
TRAIN_FILE='./train.dl.bin.txt'     #training file
TEST_FILE='./test.dl.bin.txt'       #test file
N_BATCHS=TRAIN_SIZE/BATCHSIZE       #number of batches in traing set
LR=0.01                             #learning rate
LAMBDA=0.005                        #regularization rate

# Declare Theano symbolic variables
x = T.matrix("x")
y = T.vector("y")
ww=numpy.zeros(FEATS)
for i in range(FEATS):
    if i==0:
        ww[i]=1
    elif ((i % 4)==1):
        ww[i]=1
w = theano.shared(ww, name="w")
b = theano.shared(0., name="b")
print "Initial model:"
# Construct Theano expression graph
p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))   # Probability that target = 1
prediction = p_1 #> 0.5                   # The prediction thresholded
xent = - y * T.log(p_1) - (1-y) * T.log(1-p_1) # Cross-entropy loss function
cost = xent.mean() + LAMBDA * ((w - ww) ** 2).sum()# The cost to minimize
gw, gb = T.grad(cost, [w, b])             # Compute the gradient of the cost
                                          # (we shall return to this in a
                                          # following section of this tutorial)

# Compile
train = theano.function(
          inputs=[x,y],
          outputs=[prediction, xent],updates=((w, w - LR * gw), (b, b- LR * gb )))#(b, b - LR * gb)))
predict = theano.function(inputs=[x], outputs=prediction)


#Get batch data from training set
INDEX=0
def getTrainData():
    global BATCHSIZE
    global INDEX
    array=[]
    arrayY=[]
    for i in range(INDEX+1,BATCHSIZE+INDEX+1):
        line=linecache.getline(TRAIN_FILE, i)
        if line.strip()!="":
            y=line[0:line.index(',')]
            x=line[line.index(',')+1:]
            arr=[float(xx) for xx in x.split(',')]
            array.append(arr)
            arrayY.append(int(y))
    xarray=numpy.array(array, dtype=theano.config.floatX)
    yarray=numpy.array(arrayY, dtype=numpy.int32)
    INDEX=INDEX+BATCHSIZE
    return [xarray,yarray]

#Get batch data from test set
TEST_INDEX=0
def getTextData():
    global BATCHSIZE
    global TEST_INDEX
    array=[]
    arrayY=[]
    for i in range(TEST_INDEX+1,BATCHSIZE+TEST_INDEX+1):
        line=linecache.getline(TEST_FILE, i)
        if line.strip()!="":
            y=line[0:line.index(',')]
            x=line[line.index(',')+1:]
            arr=[float(xx) for xx in x.split(',')]
            array.append(arr)
            arrayY.append(int(y))
    xarray=numpy.array(array, dtype=theano.config.floatX)
    yarray=numpy.array(arrayY, dtype=numpy.int32)
    TEST_INDEX=BATCHSIZE+TEST_INDEX
    return [xarray,yarray]

#get all test set
def getAllTextData():
    global BATCHSIZE
    global TEST_INDEX
    array=[]
    arrayY=[]
    for i in range(TEST_SIZE):
        line=linecache.getline(TEST_FILE, i)
        if line.strip()!="":
            y=line[0:line.index(',')]
            x=line[line.index(',')+1:]
            arr=[float(xx) for xx in x.split(',')]
            array.append(arr)
            arrayY.append(int(y))
    xarray=numpy.array(array, dtype=theano.config.floatX)
    yarray=numpy.array(arrayY, dtype=numpy.int32)
    return [xarray,yarray]

#get all train set
def getAllTrainData():
    global BATCHSIZE
    global TEST_INDEX
    array=[]
    arrayY=[]
    for i in range(TRAIN_SIZE):
        line=linecache.getline(TRAIN_FILE, i)
        if line.strip()!="":
            y=line[0:line.index(',')]
            x=line[line.index(',')+1:]
            arr=[float(xx) for xx in x.split(',')]
            array.append(arr)
            arrayY.append(int(y))
    xarray=numpy.array(array, dtype=theano.config.floatX)
    yarray=numpy.array(arrayY, dtype=numpy.int32)
    return [xarray,yarray]

#first prediction
x,y=getAllTextData()
y=y.tolist()
yp=(predict(numpy.asarray(x.tolist()))).tolist()
auc = roc_auc_score(y, yp)
rmse = math.sqrt(mean_squared_error(y, yp))
print '\t' + str(auc) + '\t' + str(rmse)

# Train
print "Training model:"
minerr=1
minerr_at_EPOCH=0
for i in range(EPOCHs):
    INDEX=0
    for j in range(N_BATCHS):
#        print 'j:',j
        x,y=getTrainData()
        pred, err = train(x,y)

# Error on test set
#    TEST_INDEX=0
#    err=0
#    testerr=0
#    for k in range(TEST_SIZE/BATCHSIZE):
#        x,y=getTextData()
#        pre=((predict(numpy.asarray(x.tolist()))))
#        err=err+sum(abs(y-pre))
#    testerr=(err/((TEST_SIZE/BATCHSIZE)*BATCHSIZE))
#    print 'EPOCH',i,': Test error',testerr
#    if testerr<minerr:
#        minerr=testerr
#        minerr_at_EPOCH=i

# train error
    x,y=getAllTrainData()
    y=y.tolist()
    yp=(predict(numpy.asarray(x.tolist()))).tolist()
    auc = roc_auc_score(y, yp)
    rmse = math.sqrt(mean_squared_error(y, yp))
    print 'Training Err:'+ str(i) + '\t' + str(auc) + '\t' + str(rmse)

# test error
    x,y=getAllTextData()
    y=y.tolist()
    yp=(predict(numpy.asarray(x.tolist()))).tolist()
    auc = roc_auc_score(y, yp)
    rmse = math.sqrt(mean_squared_error(y, yp))
    print 'Test Err:' + str(i) + '\t' + str(auc) + '\t' + str(rmse)
print 'minimal test error is', minerr,' at EPOCH ', minerr_at_EPOCH

#print w.get_value(), b.get_value()
#print "target values for D:", D[1]
#print "prediction on D:", predict(D[0])