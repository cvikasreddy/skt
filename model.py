
# coding: utf-8

# In[1]:

import tensorflow as tf
import numpy as np
import random
#from tqdm import tqdm # ProgressBar for loops
                                                                                                            
from tensorflow.python.ops import rnn_cell, seq2seq
from utils.data_loader import SKTDataLoader

#import os
#os.environ["CUDA_VISIBLE_DEVICES"]=""


# In[2]:

num_layers = 1   # Number of layers of RNN
num_hidden = 128 # Hidden size of RNN cell
batch_size = 64 
seq_length = 50  # Length of sequence
num_outputs = 2  
learning_rate = 0.0001
num_epochs = 10

model_name = 'skt'


# In[3]:

data_loader = SKTDataLoader('data/input_complete_split.txt', 'data/output_complete_split.txt',batch_size,seq_length)
vocab_size =  data_loader.vocab_size   # Number of unique words in dataset

data_size = data_loader.data_size      # Number of paris in the entire dataset
train_set_size = data_loader.train_size# Number of pairs in train set
valid_set_size = data_loader.valid_size# Number of pairs in valid set
test_set_size = data_loader.test_size  # Number of pairs in test set

num_train_batches = int(train_set_size*1.0/batch_size) # Number of train batches1
num_valid_batches = int(valid_set_size*1.0/batch_size)
num_test_batches = int(test_set_size*1.0/batch_size)

print "Vocab Size: " + str(vocab_size)
print "Data Size: " + str(data_size)
print train_set_size, valid_set_size, test_set_size


# In[4]:

with tf.name_scope('encode_input'):
    encode_input = [tf.placeholder(tf.int32, shape=(None,), name = "ei_%i" %i) for i in range(seq_length)]

with tf.name_scope('labels'):
    labels = [tf.placeholder(tf.int32, shape=(None,), name = "l_%i" %i) for i in range(seq_length)]

with tf.name_scope('decode_input'):
    decode_input = [tf.zeros_like(encode_input[0], dtype=np.int32, name="GO")] + labels[:-1]
    
with tf.name_scope('dropout'):
    keep_prob = tf.placeholder("float", name='keep_prob')


# In[5]:

cells = [rnn_cell.DropoutWrapper(
        rnn_cell.BasicLSTMCell(num_hidden), output_keep_prob=keep_prob
    ) for i in range(num_layers)]

stacked_lstm = rnn_cell.MultiRNNCell(cells)

with tf.variable_scope("decoders") as scope:
    decode_outputs, decode_state = seq2seq.embedding_rnn_seq2seq(
        encode_input, decode_input, stacked_lstm, vocab_size, vocab_size, 128)
    
#     decode_outputs, decode_state = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
#         encode_input, decode_input, stacked_lstm, vocab_size, vocab_size, num_hidden)
    
    scope.reuse_variables()
    
#     decode_outputs, decode_state = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
#         encode_input, decode_input, stacked_lstm, vocab_size, vocab_size, num_hidden, feed_previous=True)
    
    decode_outputs_test, decode_state_test = seq2seq.embedding_rnn_seq2seq(
        encode_input, decode_input, stacked_lstm, vocab_size, vocab_size, 128, feed_previous=True)


# In[6]:

with tf.name_scope('loss'):
    loss_weights = [tf.ones_like(l, dtype=tf.float32) for l in labels]
    loss = seq2seq.sequence_loss(decode_outputs, labels, loss_weights, vocab_size)

tf.scalar_summary('loss', loss)


# In[7]:

optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)


# In[8]:

init = tf.initialize_all_variables()
saver = tf.train.Saver()

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
#sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

sess = tf.InteractiveSession()
merged = tf.merge_all_summaries()
summary_writer = tf.train.SummaryWriter('logs/' + model_name , sess.graph)

# Waiting till gpu is available
import subprocess, time

total_memory = 12206
max_occupied = 9000

gpu_output = subprocess.check_output(["nvidia-smi"])
memory_occupied = int(gpu_output.split('MiB / ' + str(total_memory) + 'MiB')[0].split('|')[-1])

print memory_occupied
while memory_occupied > max_occupied:
    gpu_output = subprocess.check_output(["nvidia-smi"])
    memory_occupied = int(gpu_output.split('MiB / ' + str(total_memory) + 'MiB')[0].split('|')[-1])
    time.sleep(0.01)
print "GPU available"


saver.restore(sess, 'models/' + model_name)
sess.run(init)


# In[ ]:

step = 0

for epoch in range(num_epochs):
    train_losses = []
    valid_losses = []
    
    # Training on train set
    for i in range(num_train_batches):
        batch_inp, batch_outp = data_loader.next_batch()
        
        input_dict = {encode_input[t]: batch_inp[t] for t in range(seq_length)}
        input_dict.update({labels[t]: batch_outp[t] for t in range(seq_length)})
        input_dict[keep_prob] = 1.0
        
        _, loss_val, summary = sess.run([train, loss, merged], feed_dict=input_dict)
        train_losses.append(loss_val)
        
        summary_writer.add_summary(summary, step)
        step += 1
        
    # Testing on valid set
    for i in range(num_valid_batches):
        batch_inp, batch_outp = data_loader.next_batch(data_type='valid')
        
        input_dict = {encode_input[t]: batch_inp[t] for t in range(seq_length)}
        input_dict.update({labels[t]: batch_outp[t] for t in range(seq_length)})
        input_dict[keep_prob] = 1.0
        
        loss_val = sess.run(loss, feed_dict=input_dict)
        valid_losses.append(loss_val)
        
    log_txt = "Epoch: " + str(epoch) + " Steps: " + str(step) +             " train_loss: " + str(round(np.mean(train_losses), 4)) + '+' + str(round(np.std(train_losses), 2)) +             " valid_loss: " + str(round(np.mean(valid_losses), 4)) + '+' + str(round(np.std(valid_losses), 2)) 
    print log_txt
    
    f = open('log.txt', 'a')
    f.write(log_txt + '\n')
    f.close()
    
    saver.save(sess, 'models/' + model_name)


# In[ ]:

test_losses = []

# Testing on test set
for i in range(num_test_batches):
    batch_inp, batch_outp = data_loader.next_batch(data_type='test')

    input_dict = {encode_input[t]: batch_inp[t] for t in range(seq_length)}
    input_dict.update({labels[t]: batch_outp[t] for t in range(seq_length)})
    input_dict[keep_prob] = 1.0

    loss_val = sess.run(loss, feed_dict=input_dict)
    test_losses.append(loss_val)


log_txt = "Test_loss: " + str(round(np.mean(test_losses), 4)) + '+' + str(round(np.std(test_losses), 2)) 
print log_txt

f = open('log.txt', 'a')
f.write(log_txt + '\n')
f.close()


# ### Getting output and analysing

# 1. get batch data like batch_q1, batch_q2
# 2. get outputs from decode_outputs_test and argmax
# 3. use swapaxes to change it to [batch_size, seq_length] for all batch_q1, batch_q2, output
# 4. use data_loader.idx2word to generate words for all batch_q1, batch_q2, output

# In[10]:

batch_inp, batch_outp = data_loader.next_batch()

input_dict = {encode_input[t]: batch_inp[t] for t in range(seq_length)}
input_dict.update({labels[t]: batch_outp[t] for t in range(seq_length)})
input_dict[keep_prob] = 1.0

loss_val, outputs = sess.run([loss, decode_outputs_test], feed_dict = input_dict)


# In[11]:

decoded_outputs = np.array(outputs).transpose([1,0,2])
decoded_outputs = np.argmax(outputs, axis = 2)

inps = np.swapaxes(batch_inp, 0, 1)
outps = np.swapaxes(batch_outp, 0, 1)
gens = np.swapaxes(decoded_outputs, 0, 1)


# In[13]:

index = random.randint(0, batch_size-1)

inp = ''.join([data_loader.idx2word[x] for x in inps[index] if x != 14681]).replace('\xe2\x96\x81', ' ')
outp = ''.join([data_loader.idx2word[x] for x in outps[index] if x != 14681]).replace('\xe2\x96\x81', ' ')
gen = ''.join([data_loader.idx2word[x] for x in gens[index] if x != 14681]).replace('\xe2\x96\x81', ' ')


# In[14]:

print inp
print outp


# In[15]:

print gen


# #### Inputting the previous word for decoder

# In[16]:

batch_inp, batch_outp = data_loader.next_batch()

input_dict = {encode_input[t]: batch_inp[t] for t in range(seq_length)}
input_dict.update({labels[t]: batch_outp[t] for t in range(seq_length)})
input_dict[keep_prob] = 1.0

#loss_val, outputs = sess.run([loss, decode_outputs_test], feed_dict = input_dict)
loss_val, outputs = sess.run([loss, decode_outputs], feed_dict = input_dict)


# In[17]:

decoded_outputs = np.array(outputs).transpose([1,0,2])
decoded_outputs = np.argmax(outputs, axis = 2)

inps = np.swapaxes(batch_inp, 0, 1)
outps = np.swapaxes(batch_outp, 0, 1)
gens = np.swapaxes(decoded_outputs, 0, 1)


# In[18]:

index = random.randint(0, batch_size-1)

inp = ''.join([data_loader.idx2word[x] for x in inps[index] if x != 14681]).replace('\xe2\x96\x81', ' ')
outp = ''.join([data_loader.idx2word[x] for x in outps[index] if x != 14681]).replace('\xe2\x96\x81', ' ')
gen = ''.join([data_loader.idx2word[x] for x in gens[index] if x != 14681]).replace('\xe2\x96\x81', ' ')


# In[19]:

print inp
print outp


# In[20]:

print gen


# In[ ]:



