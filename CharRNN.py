import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

from preprocess import replace_unknown_chars, replace_rare_chars
from vars import MODEL_PATH, CHARS_PATH, DATA_PATH
from vars import HIDDEN_SIZE, NUM_LAYERS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(input_size, input_size)
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.decoder = nn.Linear(hidden_size, output_size)
    
    def forward(self, input_seq, hidden_state):
        embedding = self.embedding(input_seq)
        output, hidden_state = self.rnn(embedding, hidden_state)
        output = self.decoder(output)
        return output, (hidden_state[0].detach(), hidden_state[1].detach())
    
def train():
    ########### Hyperparameters ###########
    hidden_size = HIDDEN_SIZE   # size of hidden state
    seq_len = 100       # length of LSTM sequence
    num_layers = NUM_LAYERS      # num of layers in LSTM layer stack
    lr = 0.002          # learning rate
    epochs = 100        # max number of epochs
    op_seq_len = 200    # total num of characters in output test sequence
    load_chk = False    # load weights from MODEL_PATH directory to continue training

    # load the text file
    data = open(DATA_PATH, 'r', encoding='utf-8').read()
    if load_chk:
        chars = open(CHARS_PATH, 'r', encoding='utf-8').read()
        data = replace_unknown_chars(data, chars)
    else:
        data, chars = replace_rare_chars(data)
        open(CHARS_PATH, 'w', encoding='utf-8').write(chars)
    data_size, vocab_size = len(data), len(chars)
    print("----------------------------------------")
    print("Data has {} characters, {} unique".format(data_size, vocab_size))
    print("----------------------------------------")
    
    # char to index and index to char maps
    char_to_ix = { ch:i for i,ch in enumerate(chars) }
    ix_to_char = { i:ch for i,ch in enumerate(chars) }
    
    # convert data from chars to indices
    data = list(data)
    for i, ch in enumerate(data):
        data[i] = char_to_ix[ch]
    
    # data tensor on device
    data = torch.tensor(data).to(device)
    data = torch.unsqueeze(data, dim=1)
    
    # model instance
    rnn = RNN(vocab_size, vocab_size, hidden_size, num_layers).to(device)
    
    # load checkpoint if True
    if load_chk:
        rnn.load_state_dict(torch.load(MODEL_PATH))
        print("Model loaded successfully !!")
        print("----------------------------------------")
    
    # loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=lr)
    
    # training loop
    for i_epoch in range(1, epochs+1):
        
        # random starting point (1st 100 chars) from data to begin
        data_ptr = np.random.randint(100)
        n = 0
        running_loss = 0
        hidden_state = None
        
        while True:
            input_seq = data[data_ptr : data_ptr+seq_len]
            target_seq = data[data_ptr+1 : data_ptr+seq_len+1]
            
            # forward pass
            output, hidden_state = rnn(input_seq, hidden_state)
            
            # compute loss
            loss = loss_fn(torch.squeeze(output), torch.squeeze(target_seq))
            running_loss += loss.item()
            
            # compute gradients and take optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update the data pointer
            data_ptr += seq_len
            n +=1

            # if at end of data : break
            if data_ptr + seq_len + 1 > data_size:
                break

            if n % 20 == 0:
                print(f'{data_ptr} of {data_size} chars done in this epoch.')
                torch.save(rnn.state_dict(), MODEL_PATH)


        # print loss and save weights after every epoch
        print("Epoch: {0} \t Loss: {1:.8f}".format(i_epoch, running_loss/n))
        torch.save(rnn.state_dict(), MODEL_PATH)
        
        # sample / generate a text sequence after every epoch
        data_ptr = 0
        hidden_state = None
        
        # random character from data to begin
        rand_index = np.random.randint(data_size-1)
        input_seq = data[rand_index : rand_index+1]
        
        print("----------------------------------------")
        while True:
            # forward pass
            output, hidden_state = rnn(input_seq, hidden_state)
            
            # construct categorical distribution and sample a character
            output = F.softmax(torch.squeeze(output), dim=0)
            dist = Categorical(output)
            index = dist.sample()
            
            # print the sampled character
            print(ix_to_char[index.item()], end='')
            
            # next input is current output
            input_seq[0][0] = index.item()
            data_ptr += 1
            
            if data_ptr > op_seq_len:
                break
            
        print("\n----------------------------------------")
        
if __name__ == '__main__':
    train()


