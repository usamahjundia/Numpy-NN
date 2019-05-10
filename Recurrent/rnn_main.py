import numpy as np
from rnn import RNNCell
from tqdm import tqdm

def DataGen(filepath,overlap=5,batch_size=32):
    with open(filepath,'r') as f:
        filecontent = f.read()
    unique_characters = sorted(list(set(filecontent)))
    chartoint = {c:i for i,c in enumerate(unique_characters)}
    inttochar = {i:c for i,c in enumerate(unique_characters)}
    yield chartoint, inttochar
    new_content = [chartoint[i] for i in filecontent]
    seqlen = len(new_content)
    xs = np.array(new_content[:-1])
    ys = np.array(new_content[1:])
    start_index = 0
    step_size = batch_size - overlap
    while True:
        end_index = start_index + batch_size
        x_b = xs[start_index:end_index]
        y_b = ys[start_index:end_index]
        yield x_b, y_b
        start_index += step_size
        if start_index > seqlen:
            break

def num_to_onehot(num,maxval):
    to_return = np.zeros(maxval)
    to_return[num] = 1
    return to_return

input_file_path = 'input.txt'
batch_size = 32
overlap = 10

data_generator = DataGen(input_file_path,overlap=overlap,batch_size=batch_size)
cti, itc = next(data_generator)

input_dim = len(cti)
hidden_dim = 100
output_dim = len(cti)

cell = RNNCell(input_dim,hidden_dim,output_dim,10)

num_epoch = 30
learning_rate = 1e-3

for epoch in range(num_epoch):
    loss = 0
    for xs,ys in tqdm(data_generator):
        xs = np.array([num_to_onehot(i,input_dim) for i in xs])
        ys = np.array([num_to_onehot(j,input_dim) for j in ys])
        # assert xs.shape == (batch_size,input_dim), "Unexpected shape : {}".format(xs.shape)
        cell.sgd_step(xs,ys,learning_rate)
        loss += cell.calculate_loss(np.expand_dims(xs,0),np.expand_dims(ys,0))
    data_generator = DataGen(input_file_path,overlap=overlap,batch_size=batch_size)
    next(data_generator)
    print("Epoch {}\nLoss : {}".format(epoch+1,loss))