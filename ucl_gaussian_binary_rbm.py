import sys
import time
import numpy as np
import dl_utils as ut
import data_fm as fm
import graphlab as gl

rng = np.random
rng.seed(1234)

class RBM(object):
    """
    Class representing a basic restricted Boltzmann machine, with
    binary stochastic visible units and binary stochastic hidden
    units.
    """
    def __init__(self, nvis, nhid, mfvis=True, mfhid=False, initvar=0.1):
        nweights = nvis * nhid
        vb_offset = nweights
        hb_offset = nweights + nvis
        
        # One parameter matrix, with views onto it specified below.
        self.params = np.empty((nweights + nvis + nhid))

        # Weights between the hiddens and visibles
        self.weights = self.params[:vb_offset].reshape(nvis, nhid)

        # Biases on the visible units
        self.visbias = self.params[vb_offset:hb_offset]

        # Biases on the hidden units
        self.hidbias = self.params[hb_offset:]

        # Attributes for scratch arrays used during sampling.
        self._hid_states = None
        self._vis_states = None

        # Instance-specific mean field settings.
        self._mfvis = mfvis
        self._mfhid = mfhid

    @property
    def numvis(self):
        """The number of visible units (i.e. dimension of the input)."""
        return self.visbias.shape[0]

    @property
    def numhid(self):
        """The number of hidden units in this model."""
        return self.hidbias.shape[0]

    def _prepare_buffer(self, ncases, kind):
        """
        Prepare the _hid_states and _vis_states buffers for
        use for a minibatch of size `ncases`, reshaping or 
        reallocating as necessary. `kind` is one of 'hid', 'vis'.
        """
        if kind not in ['hid', 'vis']:
            raise ValueError('kind argument must be hid or vis')
        name = '_%s_states' % kind
        num = getattr(self, 'num%s' % kind)
        buf = getattr(self, name)
        if buf is None or buf.shape[0] < ncases:
            if buf is not None:
                del buf
            buf = np.empty((ncases, num))
            setattr(self, name, buf)
        buf[...] = np.NaN
        return buf[:ncases]

    def hid_activate(self, input, mf=False):
        """
        Activate the hidden units by sampling from their conditional
        distribution given each of the rows of `inputs. If `mf` is True, 
        return the deterministic, real-valued probabilities of activation
        in place of stochastic binary samples ('mean-field').
        """
        input = np.atleast_2d(input)
        ncases, ndim = input.shape
        hid = self._prepare_buffer(ncases, 'hid')
        self._update_hidden(input, hid, mf)
        return hid

    def _update_hidden(self, vis, hid, mf=False):
        """
        Update hidden units by writing new values to array `hid`.

        If `mf` is False, hidden unit values are sampled from their
        conditional distribution given the visible unit configurations
        specified in each row of `vis`. If `mf` is True, the 
        deterministic, real-valued probabilities of activation are
        written instead of stochastic binary samples ('mean-field').
        """
        hid[...] = np.dot(vis, self.weights)
        hid[...] += self.hidbias
        hid *= -1.
        np.exp(hid, hid)
        hid += 1.
        hid **= -1.
        if not mf:
            self.sample_hid(hid)
    
    def _update_visible(self, vis, hid, mf=False):
        """
        Update visible units by writing new values to array `hid`.

        If `mf` is False, visible unit values are sampled from their
        conditional distribution given the hidden unit configurations
        specified in each row of `hid`. If `mf` is True, the 
        deterministic, real-valued probabilities of activation are
        written instead of stochastic binary samples ('mean-field').
        """
        
        # Implements 1/(1 + exp(-WX) with in-place operations
        vis[...] = np.dot(hid, self.weights.T)
        vis[...] += self.visbias
        vis *= -1.
        np.exp(vis, vis)
        vis += 1.
        vis **= -1.
        if not mf:
           self.sample_vis(vis)
    
    @classmethod
    def binary_threshold(cls, probs):
        """
        Given a set of real-valued activation probabilities,
        sample binary values with the given Bernoulli parameter,
        and update the array in-placewith the Bernoulli samples.
        """
        samples = rng.uniform(size=probs.shape)
        
        # Simulate Bernoulli trials with p = probs[i,j] by generating random
        # uniform and counting any number less than probs[i,j] as success.
        probs[samples < probs] = 1.

        # Anything not set to 1 should be 0 once floored.
        np.floor(probs, probs)

    # Binary hidden units
    sample_hid = binary_threshold

    # Binary visible units
    sample_vis = binary_threshold

    def gibbs_walk(self, nsteps, hid):
        """
        Perform nsteps of alternating Gibbs sampling, 
        sampling the hidden units in parallel followed by the 
        visible units. 
        
        Depending on instantiation arguments, one or both sets of
        units may instead have "mean-field" activities computed.
        Mean-field is always used in lieu of sampling for the
        terminal hidden unit configuration.
        """
        hid = np.atleast_2d(hid)
        ncases = hid.shape[0]

        # Allocate (or reuse) a buffer with which to store 
        # the states of the visible units
        vis = self._prepare_buffer(ncases, 'vis')

        for iter in xrange(nsteps):
            
            # Update the visible units conditioning on the hidden units.
            self._update_visible(vis, hid, self._mfvis)

            # Always do mean-field on the last hidden unit update to get a
            # less noisy estimate of the negative phase correlations.
            if iter < nsteps - 1:
                mfhid = self._mfhid
            else:
                mfhid = True
            
            # Update the hidden units conditioning on the visible units.
            self._update_hidden(vis, hid, mfhid)

        return self._vis_states[:ncases], self._hid_states[:ncases]

class GaussianBinaryRBM(RBM):
    def _update_visible(self, vis, hid, mf=False):
        vis[...] = np.dot(hid, self.weights.T)
        vis += self.visbias
        if not mf:
            self.sample_vis(vis)

    @classmethod
    def sample_vis(self, vis):
        vis += rng.normal(size=vis.shape)

class CDTrainer(object):
    """An object that trains a model using vanilla contrastive divergence."""
    
    def __init__(self, model, weightcost=0.0002, rates=(1e-4, 1e-4, 1e-4),
                 cachebatchsums=True):
        self._model = model
        self._visbias_rate, self._hidbias_rate, self._weight_rate = rates
        self._weightcost = weightcost
        self._cachebatchsums = cachebatchsums
        self._weightstep = np.zeros(model.weights.shape)

    def train(self,file_path,epochs,ncases,fm_model_file,results=None,cdsteps=1, minibatch=100, momentum=0.9):
        """
        Train an RBM with contrastive divergence, using `nsteps`
        steps of alternating Gibbs sampling to draw the negative phase
        samples.
        """
        # print data
        # data = np.atleast_2d(data)
        # print data
        # index=1
        # ncases, ndim = data.shape
        model = self._model
        
        if self._cachebatchsums:
            batchsums = {}

        for epoch in xrange(epochs):

            # An epoch is a single pass through the training data.
            
            epoch_start = time.clock()
           
            # Mean squared error isn't really the right thing to measure
            # for RBMs with binary visible units, but gives a good enough
            # indication of whether things are moving in the right way.

            mse = 0
            
            # Compute the summed visible activities once

            # for offset in xrange(0, ncases, minibatch):
            o_fm=fm.DataFM(fm_model_file)
            offset=0
            while True:

                f,batch,y=o_fm.get_batch_data(file_path,(offset+1),minibatch)
                if results !=None:
                    i=0
                    for r in results:
                        i+=1
                        if i%2==1:
                            weights=r
                        else:
                            bias=r
                            batch=np.dot(batch,weights)+bias
                # batch = data[offset:(offset+minibatch)]
				batch = 1.0 / (1.0 + np.exp(-batch))
                batchsize = batch.shape[0]

                # Mean field pass on the hidden units f
                hid = model.hid_activate(batch, mf=True)
                
                # Correlations between the data and the hidden unit activations
                poscorr = np.dot(batch.T, hid)
                
                # Activities of the hidden units
                posact = hid.sum(axis=0)

                # Threshold the hidden units so that they can't convey 
                # more than 1 bit of information in the subsequent
                # sampling (assuming the hidden units are binary,
                # which they most often are).
                model.sample_hid(hid)

                # Simulate Gibbs sampling for a given number of steps.
                vis, hid = model.gibbs_walk(cdsteps, hid)

                # Update the weights with the difference in correlations
                # between the positive and negative phases.
                
                thisweightstep = poscorr
                thisweightstep -= np.dot(vis.T, hid)
                thisweightstep /= batchsize
                thisweightstep -= self._weightcost * model.weights
                thisweightstep *= self._weight_rate
               
                self._weightstep *= momentum
                self._weightstep += thisweightstep

                model.weights += self._weightstep
                
                # The gradient of the visible biases is the difference in
                # summed visible activities for the minibatch.
                if self._cachebatchsums:
                    if offset not in batchsums:
                        batchsum = batch.sum(axis=0)
                        batchsums[offset] = batchsum
                    else:
                        batchsum = batchsums[offset]
                else:
                    batchsum = batch.sum(axis=0)
                
                visbias_step = batchsum - vis.sum(axis=0)
                visbias_step *= self._visbias_rate / batchsize

                model.visbias += visbias_step

                # The gradient of the hidden biases is the difference in
                # summed hidden activities for the minibatch.

                hidbias_step = posact - hid.sum(axis=0)
                hidbias_step *= self._hidbias_rate / batchsize

                model.hidbias += hidbias_step
                
                # Compute the squared error in-place.
                vis -= batch
                vis **= 2.
                
                # Add to the total epoch estimate.
                mse += vis.sum() / ncases

                offset+=batch.shape[0]
                # print offset
                # print batch.shape[0]
                # print minibatch
                if batch.shape[0]<minibatch:
                    break

            print "Done epoch %d: %f seconds, MSE=%f" % \
                    (epoch + 1, time.clock() - epoch_start, mse)
            sys.stdout.flush()


def get_batch(file):
	f = gl.SFrame.read_csv(file, header=False, delimiter=' ')
	print f
def get_rbm_weights(file, arr, ncases, fm_model_file,batch_size=100000):
    epochs=10
    row=0
    col=0
    results=[]
    weights=[]
    bias=[]
    index=0
    for line in arr:

        index+=1
        if index==1:
            col=int(line)
        else:
            row=col
            col=int(line)
        if index==2:
            rbm = GaussianBinaryRBM(row, col)
            rbm.params[:] = rng.uniform(-1./10, 1./10, len(rbm.params))
            trainer = CDTrainer(rbm)
            trainer.train(file,epochs,ncases,minibatch=batch_size,fm_model_file=fm_model_file)
            results.append(rbm.weights)
            results.append(rbm.hidbias)
        elif index>2:
            rbm = RBM(row, col)
            rbm.params[:] = rng.uniform(-1./10, 1./10, len(rbm.params))
            trainer = CDTrainer(rbm)
            trainer.train(file, epochs,ncases,results=results,minibatch=batch_size,fm_model_file=fm_model_file)
            results.append(rbm.weights)
            results.append(rbm.hidbias)
    return results
