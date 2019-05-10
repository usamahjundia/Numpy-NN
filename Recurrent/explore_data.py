#%% 
import numpy as np
import nltk
from pprint import pprint

#%%
text_file = './input.txt'
with open(text_file,'r') as f:
    content = f.read()

#%%
print(content)

#%%
unique_characters = sorted(list(set(content)))
pprint(unique_characters)

#%%
chartoint = {c:i for i,c in enumerate(unique_characters)}
inttochar = {i:c for i,c in enumerate(unique_characters)}
pprint(chartoint)
print(len(chartoint))
pprint(inttochar)
print(len(inttochar))

#%%
x = '\n'
print(x)
print(chartoint[x])

#%%
new_content = [chartoint[i] for i in content]
print(len(new_content))

#%%
x_train = np.array(new_content[:-1])
y_train = np.array(new_content[1:])
print(x_train.shape,y_train.shape)

#%%
def dataGen(filepath,overlap=5,batch_size=32):
    with open(filepath,'r') as f:
        filecontent = f.read()
    unique_characters = sorted(list(set(filecontent)))
    chartoint = {c:i for i,c in enumerate(unique_characters)}
    inttochar = {i:c for i,c in enumerate(unique_characters)}
    yield chartoint, inttochar
    new_content = [chartoint[i] for i in content]
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