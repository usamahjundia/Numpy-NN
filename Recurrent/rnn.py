import numpy as np

class RNNCell():

    def __init__(self, input_dims, state_dims, output_dims, backprop_truncate):
        self.input_dims = input_dims
        self.state_dims = state_dims
        self.output_dims = output_dims
        self.backprop_truncate = backprop_truncate
        self.whx = np.random.uniform(-1/np.sqrt(input_dims),1/np.sqrt(input_dims),(state_dims,input_dims))
        self.whh = np.random.uniform(-1/np.sqrt(state_dims),1/np.sqrt(state_dims),(state_dims,state_dims))
        self.wyh = np.random.uniform(-1/np.sqrt(state_dims),1/np.sqrt(state_dims),(output_dims,state_dims))
    
    def softmax(self,x):
        return np.exp(x) / np.sum(np.exp(x))

    def tanh(self,x,prime=False):
        if not prime:
            return np.tanh(x)
        else:
            return 1 - np.square(np.tanh(x))

    def feedforward(self,input_seq):
        states = np.zeros((len(input_seq) + 1,self.state_dims))
        outputs = np.zeros((len(input_seq),self.output_dims))
        for t in np.arange(len(input_seq)):
            # print(self.whx.dot(input_seq[t]).shape,self.whh.dot(states[t-1,:]).shape)
            states[t,:] = self.tanh(self.whx.dot(input_seq[t]) + self.whh.dot(states[t-1,:]))
            assert states[t,:].shape == (self.state_dims,)
            outputs[t,:] = self.softmax(self.wyh.dot(states[t,:]))
            assert outputs[t,:].shape == (self.output_dims,)
        return outputs, states
    
    def _calculate_loss_individual(self,x,y):
        N = len(y)
        outputs,_ = self.feedforward(x)
        assert outputs.shape == y.shape, "Output length and label length is different"
        total_loss = 0
        for y, o in zip(y,outputs):
            total_loss += -1 * np.sum(y * np.log(o))
        return total_loss / N
    
    def calculate_loss(self,xs,ys):
        total_loss = 0
        for x,y in zip(xs,ys):
            total_loss += self._calculate_loss_individual(x,y)
        return total_loss

    ## Attempt number 5
    ## Finished!
    def backpropagation_through_time(self,x,y):
        timesteps = x.shape[0]
        outputs, states = self.feedforward(x)
        assert outputs.shape == (timesteps, self.output_dims)
        assert outputs.shape == y.shape, "Output shape is not same with y shape. Got output with {} and y with {}".format(outputs.shape,y.shape)
        assert states.shape == (timesteps+1, self.state_dims)
        dLdWyh = np.zeros_like(self.wyh)
        dLdWhh = np.zeros_like(self.whh)
        dLdWhx = np.zeros_like(self.whx)
        states_prime = 1 - states**2
        ## Gradient of error w.r.e to 
        ## the weighted sums
        ## just need to substract 1 from
        ## the output indice because its
        ## one-hot encoded
        dLdZ = outputs - y
        for timestep in np.arange(timesteps)[::-1]:
            dLdWyh += np.outer(dLdZ[timestep],states[timestep].T)
            assert dLdWyh.shape == self.wyh.shape, "Dimensions error in calculating dLdWyh. Got {} instead of {}".format(dLdWyh.shape,self.wyh.shape)
            dLdS = self.wyh.T.dot(dLdZ[timestep])
            # i grouped up the weighted sums inside the tanh into box (dont ask)
            dLdBox = dLdS * states_prime[timestep]
            for backstep in np.arange(np.max([0,timestep - self.backprop_truncate]), timestep+1)[::-1]:
                dLdWhh += np.outer(dLdBox,states[backstep-1])
                assert dLdWhh.shape == self.whh.shape, "Dimensions error in calculating dLdWhh. Got {} instead of {}".format(dLdWhh.shape,self.whh.shape)
                dLdWhx += np.outer(dLdBox,x[backstep])
                assert dLdWhx.shape == self.whx.shape, "Dimensions error in calculating dLdWhx. Got {} instead of {}".format(dLdWhx.shape,self.whx.shape)
                # update dLdBox,add terms for box in the prev time step
                dLdBox = self.whh.T.dot(dLdBox) * states_prime[backstep-1]
            
        return dLdWyh, dLdWhh, dLdWhx
    
    def sgd_step(self,x,y,learning_rate):
        dLdWyh, dLdWhh, dLdWhx = self.backpropagation_through_time(x,y)
        self.wyh -= learning_rate * dLdWyh
        self.whh -= learning_rate * dLdWhh
        self.whx -= learning_rate * dLdWhx
    
    def train_one_batch(self,xs,ys,learning_rate=1e-3):
        batchlen = len(xs)
        delta_wyh = np.zeros_like(self.wyh)
        delta_whh = np.zeros_like(self.whh)
        delta_whx = np.zeros_like(self.whx)
        for x,y in zip(xs,ys):
            print(x,y)
            dLdWyh, dLdWhh, dLdWhx = self.backpropagation_through_time(xs,ys)
            delta_wyh += 1 / batchlen * dLdWyh
            delta_whh += 1 / batchlen * dLdWhh
            delta_whx += 1 / batchlen * dLdWhx
        self.wyh -= learning_rate * delta_wyh
        self.whh -= learning_rate * delta_whh
        self.whx -= learning_rate * delta_whx

    
    def train(self,xs,ys,learning_rate=1e-5,epoch_num=30,evaluate_after=5):
        losses=[]
        loss = self.calculate_loss(xs,ys)
        losses.append(loss)
        for epoch in range(epoch_num):
            if epoch % evaluate_after == 0:
                if len(loss) > 1 and (loss[-1] > loss[-2]):
                    learning_rate *= 0.5
                pass
            for x,y in zip(xs,ys):
                self.sgd_step(x,y,learning_rate)
            print("Epoch {}\nLoss : {}".format(epoch+1,loss))


    def predict(self,input_seq):
        outputs, _ = self.feedforward(input_seq)
        return np.argmax(outputs,axis=1)