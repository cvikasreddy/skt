
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

num_layers = 3 # Number of layers of RNN
num_hidden = 128 # Hidden size of RNN cell
batch_size = 128 # Number of sentences in a batch
seq_length = 35 # Length of sequence
split = [0.9, 0.1, 0] # Splitting proportions into train, valid, test
learning_rate = 0.001 # Initial learning rate
keep_prob_val = 0.8 # keep_prob is 1 - dropout i.e., if dropout = 0.2, then keep_prob is 0.8
num_epochs = 100
verbose = 1      # Display every <verbose> epochs

model_name = 'attn_3_8000_0.8_trainonly' # Name is <num_layers>_<sentencepiece_vocabsize>_<keep_prob>


# In[3]:
data_loader = SKTDataLoader('data/dcs_data_input_train_sent.txt','data/dcs_data_output_train_sent.txt',batch_size,seq_length, split=split)
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
    decode_outputs, decode_state = seq2seq.embedding_attention_seq2seq(encode_input, decode_input, stacked_lstm, vocab_size, vocab_size, num_hidden)

    scope.reuse_variables()

    decode_outputs_test, decode_state_test = seq2seq.embedding_attention_seq2seq(encode_input, decode_input, stacked_lstm, vocab_size, vocab_size, num_hidden, feed_previous=True)
    

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

sess = tf.InteractiveSession()
merged = tf.merge_all_summaries()
summary_writer = tf.train.SummaryWriter('logs/' + model_name , sess.graph)

sess.run(init)
#saver.restore(sess, 'models/' + model_name)


# In[9]:

step = 0
try:
    for epoch in range(num_epochs):
        train_losses = []
        valid_losses = []

        # Training on train set
        for i in tqdm(range(num_train_batches)):
            batch_inp, batch_outp = data_loader.next_batch()

            input_dict = {encode_input[t]: batch_inp[t] for t in range(seq_length)}
            input_dict.update({labels[t]: batch_outp[t] for t in range(seq_length)})
            input_dict[keep_prob] = keep_prob_val

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
            log_txt = "Epoch: " + str(epoch) + " Steps: " + str(step) + " train_loss: " + str(round(np.mean(train_losses),4)) + '+' + str(round(np.std(train_losses),2)) +                 " valid_loss: " + str(round(np.mean(valid_losses),4)) + '+' + str(round(np.std(valid_losses),2)) 
            print log_txt

            f = open('log.txt', 'a')
            f.write(log_txt + '\n')
            f.close()

            saver.save(sess, 'models/' + model_name)
except KeyboardInterrupt:
    print "Stopped at epoch: " + str(epoch) + ' and step: ' + str(step)

print "Training completed"