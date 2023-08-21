---
# file: _projects/TV Script Generation.md
layout:      project
title:       TV Script Generation*
date:        10 May 2021
image:
  path:       /assets/projects/tvscript.jpg
  srcset:
    1920w:   /assets/projects/tvscript.jpg
    960w:    /assets/projects/tvscript.jpg
    480w:    /assets/projects/tvscript.jpg
# caption:     Hyde is a brazen two-column Jekyll theme.
description: >
  In this project, we will generate your own Seinfeld TV scripts using RNNs. We will be using part of the Seinfeld dataset of scripts from 9 seasons.
   The Neural Network we will build will generate a new, "fake" TV script, based on patterns it recognizes in this training data.
#links:
#  - title:   Demo
#    url:     _posts/2021-05-10-tv_script_generation.md
#  - title:   Source
#    url:     https://github.com/Zakaria-Alsahfi/zakariaalsahfi.github.io/blob/cfb388cf10ec30f210c6b4699678e7da5a6ce8da/_posts/2021-05-10-tv_script_generation.md
featured:    false
tags: [Data Science, Machine Learning, Deep Learning, Text Mining, Python]
---

## Get the Data

```python
# load in data
import helper
data_dir = './data/Seinfeld_Scripts.txt'
text = helper.load_data(data_dir)
```

## Explore the Data
Play around with `view_line_range` to view different parts of the data. This will give you a sense of the data you'll be working with. You can see, for example, that it is all lowercase text, and each new line of dialogue is separated by a newline character `\n`.


```python
view_line_range = (0, 10)

print('Dataset Stats')
print('Roughly the number of unique words: {}'.format(len({word: None for word in text.split()})))

lines = text.split('\n')
print('Number of lines: {}'.format(len(lines)))
word_count_line = [len(line.split()) for line in lines]
print('Average number of words in each line: {}'.format(np.average(word_count_line)))

print()
print('The lines {} to {}:'.format(*view_line_range))
print('\n'.join(text.split('\n')[view_line_range[0]:view_line_range[1]]))
```

    Dataset Stats
    Roughly the number of unique words: 46367
    Number of lines: 109233
    Average number of words in each line: 5.544240293684143
    
    The lines 0 to 10:
    jerry: do you know what this is all about? do you know, why were here? to be out, this is out...and out is one of the single most enjoyable experiences of life. people...did you ever hear people talking about we should go out? this is what theyre talking about...this whole thing, were all out now, no one is home. not one person here is home, were all out! there are people trying to find us, they dont know where we are. (on an imaginary phone) did you ring?, i cant find him. where did he go? he didnt tell me where he was going. he must have gone out. you wanna go out you get ready, you pick out the clothes, right? you take the shower, you get all ready, get the cash, get your friends, the car, the spot, the reservation...then youre standing around, what do you do? you go we gotta be getting back. once youre out, you wanna get back! you wanna go to sleep, you wanna get up, you wanna go out again tomorrow, right? where ever you are in life, its my feeling, youve gotta go. 
    
    jerry: (pointing at georges shirt) see, to me, that button is in the worst possible spot. the second button literally makes or breaks the shirt, look at it. its too high! its in no-mans-land. you look like you live with your mother. 
    
    george: are you through? 
    
    jerry: you do of course try on, when you buy? 
    
    george: yes, it was purple, i liked it, i dont actually recall considering the buttons. 
    


---
## Implement Pre-processing Functions
The first thing to do to any dataset is pre-processing.  Implement the following pre-processing functions below:
- Lookup Table
- Tokenize Punctuation

### Lookup Table
To create a word embedding, you first need to transform the words to ids.  In this function, create two dictionaries:
- Dictionary to go from the words to an id, we'll call `vocab_to_int`
- Dictionary to go from the id to word, we'll call `int_to_vocab`

Return these dictionaries in the following **tuple** `(vocab_to_int, int_to_vocab)`


```python
def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    # TODO: Implement Function
    words_counts = Counter(text) # Extract all unique words and their respective frequency
    unique_words = sorted(words_counts, key=words_counts.get, reverse=True) # Extract all unique words only
    
    # Create vocab_to_int and int_to_vocab dicts
    int_to_vocab = {ii: word for ii, word in enumerate(unique_words)}
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}

    # return tuple
    return (vocab_to_int, int_to_vocab)

tests.test_create_lookup_tables(create_lookup_tables)
```

    Tests Passed


### Tokenize Punctuation
We'll be splitting the script into a word array using spaces as delimiters. However, punctuations like periods and exclamation marks can create multiple ids for the same word. For example, "bye" and "bye!" would generate two different word ids.

This dictionary will be used to tokenize the symbols and add the delimiter (space) around it.  This separates each symbols as its own word, making it easier for the neural network to predict the next word. Make sure you don't use a value that could be confused as a word; for example, instead of using the value "dash", try using something like "||dash||".


```python
def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenized dictionary where the key is the punctuation and the value is the token
    """
    # TODO: Implement Function
    tokens = {
    '.' : '||Period||',
    ',' : '||Comma||',
    '"' : '||Quotation_Mark||',
    ';' : '||Semicolon||',
    '!' : '||Exclamation_Mark||',
    '?' : '||Question_Mark||',
    '(' : '||Left_Parentheses||',
    ')' : '||Right_Parentheses||',
    '-' : '||Dash||',
    '\n' : '||Return||'
    }
        
    return tokens

tests.test_tokenize(token_lookup)
```

    Tests Passed


## Pre-process all the data and save it

```python
# pre-process training data
helper.preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)
```

# Check Point

```python
int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
```

## Build the Neural Network
In this section, we will build the components necessary to build an RNN by implementing the RNN Module and forward and backpropagation functions.

### Check Access to GPU


```python
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('No GPU found. Please use a GPU to train your neural network.')
```

## Input
Let's start with the preprocessed input data. We'll use [TensorDataset](http://pytorch.org/docs/master/data.html#torch.utils.data.TensorDataset) to provide a known format to our dataset; in combination with [DataLoader](http://pytorch.org/docs/master/data.html#torch.utils.data.DataLoader), it will handle batching, shuffling, and other dataset iteration functions.

We can create data with TensorDataset by passing in feature and target tensors. Then create a DataLoader as usual.
```
data = TensorDataset(feature_tensors, target_tensors)
data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size)
```

### Batching
Implement the `batch_data` function to batch `words` data into chunks of size `batch_size` using the `TensorDataset` and `DataLoader` classes.

```python
from torch.utils.data import TensorDataset, DataLoader


def batch_data(words, sequence_length, batch_size):
    """
    Batch the neural network data using DataLoader
    :param words: The word ids of the TV scripts
    :param sequence_length: The sequence length of each batch
    :param batch_size: The size of each batch; the number of sequences in a batch
    :return: DataLoader with batched data
    """
    # TODO: Implement function

    n_batches = len(words) - sequence_length
    words = words
    feature_tensor, target_tensor = [], []
    
    for idx in range(0, n_batches):
        feature_tensor.append(words[idx:idx+sequence_length])
        
        target_tensor.append(words[idx+sequence_length])
    
    data = TensorDataset(torch.from_numpy(np.asarray(feature_tensor)), 
                         torch.from_numpy(np.asarray(target_tensor)))
    data_loader = DataLoader(data, batch_size=batch_size)
    
    return data_loader
```

### Test dataloader 

```python
# test dataloader

test_text = range(50)
t_loader = batch_data(test_text, sequence_length=5, batch_size=10)

data_iter = iter(t_loader)
sample_x, sample_y = data_iter.next()

print(sample_x.shape)
print(sample_x)
print()
print(sample_y.shape)
print(sample_y)
```

    torch.Size([10, 5])
    tensor([[  0,   1,   2,   3,   4],
            [  1,   2,   3,   4,   5],
            [  2,   3,   4,   5,   6],
            [  3,   4,   5,   6,   7],
            [  4,   5,   6,   7,   8],
            [  5,   6,   7,   8,   9],
            [  6,   7,   8,   9,  10],
            [  7,   8,   9,  10,  11],
            [  8,   9,  10,  11,  12],
            [  9,  10,  11,  12,  13]])
    
    torch.Size([10])
    tensor([  5,   6,   7,   8,   9,  10,  11,  12,  13,  14])

## Build the Neural Network
Implement an RNN using PyTorch's [Module class](http://pytorch.org/docs/master/nn.html#torch.nn.Module).

The initialize function should create the layers of the neural network and save them to the class. The forward propagation function will use these layers to run forward propagation and generate an output and a hidden state.

```python
class RNN(nn.Module):
    
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5):
        """
        Initialize the PyTorch RNN Module
        :param vocab_size: The number of input dimensions of the neural network (the size of the vocabulary)
        :param output_size: The number of output dimensions of the neural network
        :param embedding_dim: The size of embeddings, should you choose to use them        
        :param hidden_dim: The size of the hidden layer outputs
        :param dropout: dropout to add in between LSTM/GRU layers
        """
        super(RNN, self).__init__()
        # TODO: Implement function
        
        # set class variables
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        # define model layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.lstm = nn.LSTM(input_size = embedding_dim, hidden_size = hidden_dim,
                            num_layers = n_layers, dropout=dropout, 
                            batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)
    
    def forward(self, nn_input, hidden):
        """
        Forward propagation of the neural network
        :param nn_input: The input to the neural network
        :param hidden: The hidden state        
        :return: Two Tensors, the output of the neural network and the latest hidden state
        """
        # TODO: Implement function
        batch_size = nn_input.size(0)
        # embeddings and lstm_out
        embeds = self.embedding(nn_input)
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        
        # fully-connected layer
        output = self.fc(lstm_out)
        
        # reshape into (batch_size, seq_length, output_size)
        out = output.view(batch_size, -1, self.output_size)
        
        # get last batch
        output = out[:, -1]
        
        # return one batch of output word scores and the hidden state
        return output, hidden
    
    def init_hidden(self, batch_size):
        '''
        Initialize the hidden state of an LSTM/GRU
        :param batch_size: The batch_size of the hidden state
        :return: hidden state of dims (n_layers, batch_size, hidden_dim)
        '''
        # Implement function
        
        # initialize hidden state with zero weights, and move to GPU if available
        weight = next(self.parameters()).data
        
        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        
        return hidden

tests.test_rnn(RNN, train_on_gpu)
```

    Tests Passed


### Define forward and backpropagation

Use the RNN class we implemented to apply forward and back propagation. 

This function should return the average loss over a batch and the hidden state returned by a call to `RNN(inp, hidden)`. Recall that you can get this loss by computing it, as usual, and calling `loss.item()`.

```python
def forward_back_prop(rnn, optimizer, criterion, inp, target, hidden):
    """
    Forward and backward propagation on the neural network
    :param rnn: The PyTorch Module that holds the neural network
    :param optimizer: The PyTorch optimizer for the neural network
    :param criterion: The PyTorch loss function
    :param inp: A batch of input to the neural network
    :param target: The target output for the batch of input
    :return: The loss and the latest hidden state Tensor
    """
    
    # TODO: Implement Function
    
    # move data to GPU, if available
    if train_on_gpu:
        inp, target = inp.cuda(), target.cuda()
        
    hidden = tuple([each.data for each in hidden])
    
    rnn.zero_grad()
    
    output, hidden = rnn(inp, hidden)
    # perform backpropagation and optimization
    # calculate loss and perform backpropagation
    loss = criterion(output, target)
    loss.backward()
    
    # use 'clip_grad_norm' to help prevent the exploding gradient problem
    nn.utils.clip_grad_norm_(rnn.parameters(), 5)
    
    optimizer.step()

    # return the loss over a batch and the hidden state produced by our model
    return loss.item(), hidden

tests.test_forward_back_prop(RNN, forward_back_prop, train_on_gpu)
```

    Tests Passed


## Neural Network Training

With the structure of the network complete and data ready to be fed in the neural network, it's time to train it.

### Train Loop

The training loop is implemented for you in the `train_decoder` function. This function will train the network over all the batches for the number of epochs given. The model progress will be shown every number of batches. This number is set with the `show_every_n_batches` parameter. You'll set this parameter along with other parameters in the next section.


```python
def train_rnn(rnn, batch_size, optimizer, criterion, n_epochs, show_every_n_batches=100):
    batch_losses = []
    
    rnn.train()

    print("Training for %d epoch(s)..." % n_epochs)
    for epoch_i in range(1, n_epochs + 1):
        
        # initialize hidden state
        hidden = rnn.init_hidden(batch_size)
        
        for batch_i, (inputs, labels) in enumerate(train_loader, 1):
            
            # make sure you iterate over completely full batches, only
            n_batches = len(train_loader.dataset)//batch_size
            if(batch_i > n_batches):
                break
            
            # forward, back prop
            loss, hidden = forward_back_prop(rnn, optimizer, criterion, inputs, labels, hidden)          
            # record loss
            batch_losses.append(loss)

            # printing loss stats
            if batch_i % show_every_n_batches == 0:
                print('Epoch: {:>4}/{:<4}  Loss: {}\n'.format(
                    epoch_i, n_epochs, np.average(batch_losses)))
                batch_losses = []

    # returns a trained rnn
    return rnn
```


```python
# Data params
sequence_length = 10   # of words in a sequence
batch_size = 256

train_loader = batch_data(int_text, sequence_length, batch_size)
```


```python
# Training parameters
num_epochs = 20
learning_rate = 0.001
vocab_size = len(vocab_to_int)
output_size = vocab_size
embedding_dim = 300
hidden_dim = 256
n_layers = 2
show_every_n_batches = 500
```

### Train
In the next cell, we will train the neural network on the pre-processed data.

> **We should aim for a loss less than 3.5.** 

```python

# create model and move to gpu if available
rnn = RNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5)
if train_on_gpu:
    rnn.cuda()

# defining loss and optimization functions for training
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# training the model
trained_rnn = train_rnn(rnn, batch_size, optimizer, criterion, num_epochs, show_every_n_batches)

# saving the trained model
helper.save_model('./save/trained_rnn', trained_rnn)
print('Model Trained and Saved')
```

    Training for 20 epoch(s)...
    Epoch:    1/20    Loss: 5.462565821170807
    
    Epoch:    1/20    Loss: 4.756056294918061
    
    Epoch:    1/20    Loss: 4.691307275772095
    
    Epoch:    1/20    Loss: 4.518818720817566
    
    Epoch:    1/20    Loss: 4.373862658977509
    
    Epoch:    1/20    Loss: 4.44727063703537
    
    Epoch:    2/20    Loss: 4.268465927945889
    
    Epoch:    2/20    Loss: 3.995042175769806
    
    Epoch:    2/20    Loss: 4.06786646604538
    
    Epoch:    2/20    Loss: 3.9948999490737913
    
    Epoch:    2/20    Loss: 3.941266675949097
    
    Epoch:    2/20    Loss: 4.041490489959717
    
    Epoch:    3/20    Loss: 3.9453085132730688
    
    Epoch:    3/20    Loss: 3.770205316543579
    
    Epoch:    3/20    Loss: 3.8474122133255007
    
    Epoch:    3/20    Loss: 3.7906584916114805
    
    Epoch:    3/20    Loss: 3.7366869487762453
    
    Epoch:    3/20    Loss: 3.8387298827171326
    
    Epoch:    4/20    Loss: 3.7503895786235004
    
    Epoch:    4/20    Loss: 3.614983685493469
    
    Epoch:    4/20    Loss: 3.6814241523742677
    
    Epoch:    4/20    Loss: 3.6458065247535707
    
    Epoch:    4/20    Loss: 3.5965149807929992
    
    Epoch:    4/20    Loss: 3.6770475759506227
    
    Epoch:    5/20    Loss: 3.6111995296749644
    
    Epoch:    5/20    Loss: 3.4932042083740233
    
    Epoch:    5/20    Loss: 3.550447774887085
    
    Epoch:    5/20    Loss: 3.5313276381492615
    
    Epoch:    5/20    Loss: 3.4760674366950988
    
    Epoch:    5/20    Loss: 3.5590632767677306
    ..........................................
    ..........................................
    ..........................................
    
    Epoch:   20/20    Loss: 2.9684349485044557
    
    Epoch:   20/20    Loss: 2.93396382522583
    
    Epoch:   20/20    Loss: 2.933423617362976
    
    Epoch:   20/20    Loss: 2.954867678165436
    
    Epoch:   20/20    Loss: 2.9034318995475767
    
    Epoch:   20/20    Loss: 2.972546745300293
  
    Model Trained and Saved


Before I trained, I realized that 200 sequence length would be far too many for a TV script. I cut that down to about 10. With a smaller sequence length, a larger batch size was possible. I chose 256 as a batch size.

Once I have trained it, I adjusted the values and keep any that have better results. I lowered the learning rate from 0.01 to 0.001, as well as decreased the embedding dimensions from 400 to 300. Also, I saw more improvements in the loss by epoch 4, so I increased the number of epochs to 20.

With these changes, the loss became less than 3.5, which I was happy with.

# Checkpoint

After running the above training cell, our model will be saved by name, `trained_rnn`, and now we will load in our word:id dictionaries _and_ load in our saved model by name!

```python
_, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
trained_rnn = helper.load_model('./save/trained_rnn')
```

## Generate TV Script
With the network trained and saved, we will use it to generate a new, "fake" Seinfeld TV script in this section.

### Generate Text
To generate the text, the network needs to start with a single word and repeat its predictions until it reaches a set length. We will be using the `generate` function to do this. It takes a word id to start with, `prime_id`, and generates a set length of text, `predict_len`. Also note that it uses topk sampling to introduce some randomness in choosing the most likely next word, given an output set of word scores!


```python

def generate(rnn, prime_id, int_to_vocab, token_dict, pad_value, predict_len=100):
    """
    Generate text using the neural network
    :param decoder: The PyTorch Module that holds the trained neural network
    :param prime_id: The word id to start the first prediction
    :param int_to_vocab: Dict of word id keys to word values
    :param token_dict: Dict of puncuation tokens keys to puncuation values
    :param pad_value: The value used to pad a sequence
    :param predict_len: The length of text to generate
    :return: The generated text
    """
    rnn.eval()
    
    # create a sequence (batch_size=1) with the prime_id
    current_seq = np.full((1, sequence_length), pad_value)
    current_seq[-1][-1] = prime_id
    predicted = [int_to_vocab[prime_id]]
    
    for _ in range(predict_len):
        if train_on_gpu:
            current_seq = torch.LongTensor(current_seq).cuda()
        else:
            current_seq = torch.LongTensor(current_seq)
        
        # initialize the hidden state
        hidden = rnn.init_hidden(current_seq.size(0))
        
        # get the output of the rnn
        output, _ = rnn(current_seq, hidden)
        
        # get the next word probabilities
        p = F.softmax(output, dim=1).data
        if(train_on_gpu):
            p = p.cpu() # move to cpu
         
        # use top_k sampling to get the index of the next word
        top_k = 5
        p, top_i = p.topk(top_k)
        top_i = top_i.numpy().squeeze()
        
        # select the likely next word index with some element of randomness
        p = p.numpy().squeeze()
        word_i = np.random.choice(top_i, p=p/p.sum())
        
        # retrieve that word from the dictionary
        word = int_to_vocab[word_i]
        predicted.append(word)     
        
        # the generated word becomes the next "current sequence" and the cycle can continue
        current_seq = np.roll(current_seq, -1, 1)
        current_seq[-1][-1] = word_i
    
    gen_sentences = ' '.join(predicted)
    
    # Replace punctuation tokens
    for key, token in token_dict.items():
        ending = ' ' if key in ['\n', '(', '"'] else ''
        gen_sentences = gen_sentences.replace(' ' + token.lower(), key)
    gen_sentences = gen_sentences.replace('\n ', '\n')
    gen_sentences = gen_sentences.replace('( ', '(')
    
    # return all the sentences
    return gen_sentences
```

### Generate a New Script
It's time to generate the text. 

```python
gen_length = 400
prime_word = 'elaine'

pad_word = helper.SPECIAL_WORDS['PADDING']
generated_script = generate(trained_rnn, vocab_to_int[prime_word + ':'], int_to_vocab, token_dict, vocab_to_int[pad_word], gen_length)
print(generated_script)
```

    elaine: i think we could go on the street.
    
    jerry: i think we can get it.
    
    jerry: you know i don't think so bad.
    
    jerry: you know what? the whole thing is going to cost.
    
    kramer: yeah, yeah, well, what did i do?
    
    kramer: i don't know.
    
    george: i know.
    
    elaine: oh, no- no, no- no! i- i was just trying to attend it.
    
    kramer: yeah, sure, i know.
    
    jerry: i don't think so.
    
    jerry: you can't believe what i'm doing?
    
    jerry: you know, the female elevator was an accident.
    
    elaine: no. no, not.
    
    jerry: no no no! no, no. i don't know.
    
    elaine: what is that?
    
    george: nothing.
    
    jerry: i don't want to hear about that.
    
    jerry: i thought 646 was gonna be prosecuted.
    
    kramer: oh, i don't believe you, jackie, i was wondering if we were tutoring the cardigan.
    
    jerry: oh, hi!
    
    elaine: oh, hi mom.
    
    elaine: what happened to the defendants?
    
    kramer: yeah, yeah.
    
    george: well, i think i could have a little chat to this jury. geraldo, you know, you have to go to california, you want to go to a hospital in the shower coolers.
    
    jerry:(thinking) oh, i think this is the only way to find the warrior princess to the general, the boring valley, but i have no idea how you appreciate it!
    
    jerry: i don't think so.
    
    george: well, it was a good mistake.
    
    jerry: no.
    
    hoyt: so you have to admit this?
    
    jerry: i don't think so.
    
    elaine: oh my god, it's a good time!
    
    george: yeah.
    
    elaine: what are we supposed to say?
    
    elaine: i don't


# The TV Script is Not Perfect
It's ok if the TV script doesn't make perfect sense.
You can see that there are multiple characters that say (somewhat) complete sentences, but it doesn't have to be perfect! It takes quite a while to get good results, and often, we will have to use a smaller vocabulary (and discard uncommon words), or get more data.  The Seinfeld dataset is about 3.4 MB, which is big enough for our purposes; for script generation we will want more than 1 MB of text, generally.
