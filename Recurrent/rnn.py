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

    def feedforward(self,input_seq):
        states = np.zeros((len(input_seq) + 1,self.state_dims))
        outputs = np.zeros((len(input_seq),self.output_dims))
        for t in np.arange(len(input_seq)):
            states[t,:] = np.tanh(self.whx.dot(input_seq[t]) + self.whh.dot(states[t-1,:]))
            outputs[t,:] = self.softmax(self.wyh.dot(states[t,:]))
        return outputs, states
    
    ## Attempt number 5
    def backpropagation_through_time(self,x,y):
        timesteps = x.shape[0]
        outputs, states = self.feedforward(x)
        dLdWyh = np.zeros_like(self.wyh)
        dLdWhh = np.zeros_like(self.whh)
        dLdWhx = np.zeros_like(self.whx)
        ## Gradient of error w.r.e to 
        ## the weighted sums
        ## just need to substract 1 from
        ## the output indice because its
        ## one-hot encoded
        outputs[:,y] -= 1
        for timestep in np.arange(timesteps)[::-1]:
            dLdWyh += np.outer(outputs[timestep],states[timestep])
            
        return dLdWyh, dLdWhh, dLdWhx

    def predict(self,input_seq):
        outputs, _ = self.feedforward(input_seq)
        return np.argmax(outputs,axis=1)