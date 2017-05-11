
# coding: utf-8

# In[1]:

import tensorflow as tf
import numpy as np
import random
from tqdm import tqdm # ProgressBar for loops
                                                                                                            
from tensorflow.python.ops import rnn_cell, seq2seq
from utils.data_loader import SKTDataLoader

#import os
#os.environ["CUDA_VISIBLE_DEVICES"]=""


# In[2]:

num_layers = 3   # Number of layers of RNN
num_hidden = 128  # Hidden size of RNN cell
batch_size = 64  # Number of sentences in a batch
seq_length = 50  # Length of sequence
split = [0.90, 0.05, 0.05] # Splitting proportions into train, valid, test
learning_rate = 0.001
num_epochs = 30
verbose = 1      # Display every <verbose> epochs

model_name = 'skt_3_1000' # Name is <skt>_<num_layers>_<sentencepiece_vocabsize>


# In[3]:

data_loader = SKTDataLoader('data/input_complete_1000_split.txt','data/output_complete_1000_split.txt',batch_size,seq_length, split=split)
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
        encode_input, decode_input, stacked_lstm, vocab_size, vocab_size, embedding_size=num_hidden)
    
#     decode_outputs, decode_state = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
#         encode_input, decode_input, stacked_lstm, vocab_size, vocab_size, num_hidden)
    
    scope.reuse_variables()
    
#     decode_outputs, decode_state = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
#         encode_input, decode_input, stacked_lstm, vocab_size, vocab_size, num_hidden, feed_previous=True)
    
    decode_outputs_test, decode_state_test = seq2seq.embedding_rnn_seq2seq(
        encode_input, decode_input, stacked_lstm,vocab_size,vocab_size,embedding_size=num_hidden,feed_previous=True)


# In[6]:

with tf.name_scope('loss'):
    loss_weights = [tf.ones_like(l, dtype=tf.float32) for l in labels]
    loss = seq2seq.sequence_loss(decode_outputs, labels, loss_weights, vocab_size)

#with tf.name_scope('accuracy'):
    #macro_accuracy = 

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

#saver.restore(sess, 'models/' + model_name)
sess.run(init)


# In[9]:

step = 0
try:
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

        if epoch % verbose == 0:
            log_txt = "Epoch: " + str(epoch) + " Steps: " + str(step) +                 " train_loss: " + str(round(np.mean(train_losses),4)) + '+' + str(round(np.std(train_losses),2)) +                 " valid_loss: " + str(round(np.mean(valid_losses),4)) + '+' + str(round(np.std(valid_losses),2)) 
            print log_txt

            f = open('log.txt', 'a')
            f.write(log_txt + '\n')
            f.close()

            saver.save(sess, 'models/' + model_name)
except KeyboardInterrupt:
    print "Stopped at epoch: " + str(epoch) + ' and step: ' + str(step)


# In[10]:

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

# 1. get batch data like batch_inp, batch_outp
# 2. get outputs from decode_outputs_test and argmax
# 3. use swapaxes to change it to [batch_size, seq_length] for all batch_inp, batch_outp, output
# 4. use data_loader.idx2word to generate words for all batch_inp, batch_outp, output

# In[13]:

batch_inp, batch_outp = data_loader.random_batch()

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


# In[12]:

#index = random.randint(0, batch_size-1)
for index in range(10):
    inp = ''.join([data_loader.idx2word[x] for x in inps[index] if x != vocab_size-1][::-1])#.replace('\xe2\x96\x81', ' ')
    outp = ''.join([data_loader.idx2word[x] for x in outps[index] if x != vocab_size-1])#.replace('\xe2\x96\x81', ' ')
    gen = ''.join([data_loader.idx2word[x] for x in gens[index] if x != vocab_size-1])#.replace('\xe2\x96\x81', ' ')
    
    inp_raw = inps[index]
    outp_raw = outps[index]
    gen_raw = gens[index]
    
    
    print inp
    print outp
    print gen
    
#     print [x for x in inp_raw[::-1] if x != vocab_size-1]
#     print [x for x in outp_raw if x != vocab_size-1]
#     print [x for x in gen_raw if x != vocab_size-1]
    print 


# #### Inputting the previous word for decoder

# In[13]:

batch_inp, batch_outp = data_loader.random_batch()

input_dict = {encode_input[t]: batch_inp[t] for t in range(seq_length)}
input_dict.update({labels[t]: batch_outp[t] for t in range(seq_length)})
input_dict[keep_prob] = 1.0

#loss_val, outputs = sess.run([loss, decode_outputs_test], feed_dict = input_dict)
loss_val, outputs = sess.run([loss, decode_outputs], feed_dict = input_dict)


# In[14]:

decoded_outputs = np.array(outputs).transpose([1,0,2])
decoded_outputs = np.argmax(outputs, axis = 2)

inps = np.swapaxes(batch_inp, 0, 1)
outps = np.swapaxes(batch_outp, 0, 1)
gens = np.swapaxes(decoded_outputs, 0, 1)


# In[15]:

#index = random.randint(0, batch_size-1)
for index in range(10):
    inp = ''.join([data_loader.idx2word[x] for x in inps[index] if x != vocab_size-1][::-1])#.replace('\xe2\x96\x81', ' ')
    outp = ''.join([data_loader.idx2word[x] for x in outps[index] if x != vocab_size-1][::-1])#.replace('\xe2\x96\x81', ' ')
    gen = ''.join([data_loader.idx2word[x] for x in gens[index] if x != vocab_size-1][::-1])#.replace('\xe2\x96\x81', ' ')
    
    print inp
    print outp
    print gen
    print


# #### Checking for test data

# In[16]:

batch_inp, batch_outp = data_loader.random_batch(data_type='test')

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


# In[20]:

#index = random.randint(0, batch_size-1)
for index in range(10):
    inp = ''.join([data_loader.idx2word[x] for x in inps[index] if x != vocab_size-1][::-1])#.replace('\xe2\x96\x81', ' ')
    outp = ''.join([data_loader.idx2word[x] for x in outps[index] if x != vocab_size-1])#.replace('\xe2\x96\x81', ' ')
    gen = ''.join([data_loader.idx2word[x] for x in gens[index] if x != vocab_size-1])#.replace('\xe2\x96\x81', ' ')
    
    inp_raw = inps[index]
    outp_raw = outps[index]
    gen_raw = gens[index]
       
    print inp
    print outp
    print gen
    
#     print [x for x in inp_raw[::-1] if x != vocab_size-1]
#     print [x for x in outp_raw if x != vocab_size-1]
#     print [x for x in gen_raw if x != vocab_size-1]
    print


# ### Calculating precision, recall

# #### Getting outputs on entire test set

# In[ ]:

data_loader.reset_index(data_type='test')

X_test = []
y_test = []
y_out = []

for i in range(num_test_batches):
    
    batch_inp, batch_outp = data_loader.next_batch(data_type='test')

    input_dict = {encode_input[t]: batch_inp[t] for t in range(seq_length)}
    input_dict.update({labels[t]: batch_outp[t] for t in range(seq_length)})
    input_dict[keep_prob] = 1.0

    loss_val, outputs = sess.run([loss, decode_outputs_test], feed_dict = input_dict)

    decoded_outputs = np.array(outputs).transpose([1,0,2])
    decoded_outputs = np.argmax(outputs, axis = 2)

    inps = np.swapaxes(batch_inp, 0, 1)
    outps = np.swapaxes(batch_outp, 0, 1)
    gens = np.swapaxes(decoded_outputs, 0, 1)

    for index in range(batch_size):
        inp = ''.join([data_loader.idx2word[x] for x in inps[index] if x != vocab_size-1][::-1])
        outp = ''.join([data_loader.idx2word[x] for x in outps[index] if x != vocab_size-1])
        gen = ''.join([data_loader.idx2word[x] for x in gens[index] if x != vocab_size-1])

        X_test.append(inp.split())
        y_test.append(outp.split())
        y_out.append(gen.split())


# In[ ]:

precisions = []
recalls = []

for outp, gen in zip(y_test, y_out):
    intersection = set(outp).intersection(gen)
    prec = intersection*1.0/len(gen)
    recall = intersection*1.0/len(outp)
    
    precisions.append(prec)
    recalls.append(recall)


# In[ ]:

avg_prec = np.mean(precisions)*100.0
avg_recall = np.mean(recalls)*100.0
f1_score = 2*avg_prec*avg_recall/(avg_prec + avg_recall)


# In[ ]:

print "Precision: " + str(avg_prec) 
print "Recall: " + str(avg_recall)
print "F1_score: " + str(f1_score)

