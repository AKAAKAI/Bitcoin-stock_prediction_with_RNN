import time
import datetime
import numpy as np
import csv
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import datetime


window_len = 10

file = open('stock_data/stock_price_direction_all_new2.csv', 'r')
csvCursor_r = csv.reader(file)
file_num = open('stock_data/stock_price_direction_all_new2.csv', 'r')
csvCursor_r_num = (len(list(csv.reader(file_num)))-1)

print(csvCursor_r_num)

inputs_tr = []
inputs_t = []
LSTM_training_inputs = []
LSTM_training_outputs = []
LSTM_test_inputs = []
LSTM_test_outputs = []

j = 0
for temp_set in csvCursor_r:
    if j == 0:
        pass
    elif j <= ( csvCursor_r_num*0.75):
        inputs_tr.append(temp_set[5:])
    else:
        inputs_t.append(temp_set[5:])
    j = j + 1

i = 0
for temp_set in inputs_tr:    
    if i < (window_len-1):
        pass
    else:
        temp_array=[]
        for j in range(i-window_len+1,i):
            temp_array.append(inputs_tr[j][:-1])
        LSTM_training_inputs.append(temp_array) 
        LSTM_training_outputs.append(temp_set[4])
    i = i + 1


i = 0
for temp_set in inputs_t:    
    if i < (window_len-1):
        pass
    else:
        temp_array=[]
        for j in range(i-window_len+1,i):
            temp_array.append(inputs_t[j][:-1])
        LSTM_test_inputs.append(temp_array) 
        LSTM_test_outputs.append(temp_set[4])
    i = i + 1


# I find it easier to work with numpy arrays rather than pandas dataframes
# especially as we now only have numerical data
LSTM_training_inputs = [np.array(LSTM_training_input).astype(np.float) for LSTM_training_input in LSTM_training_inputs]
LSTM_training_inputs = np.array(LSTM_training_inputs)

LSTM_training_outputs = [np.array(list(LSTM_training_output)[1:]) for LSTM_training_output in LSTM_training_outputs]
LSTM_training_outputs = np.array(LSTM_training_outputs)



LSTM_test_inputs = [np.array(LSTM_test_inputs).astype(np.float) for LSTM_test_inputs in LSTM_test_inputs]
LSTM_test_inputs = np.array(LSTM_test_inputs)

LSTM_test_outputs = [np.array(list(LSTM_test_output)[1:]) for LSTM_test_output in LSTM_test_outputs]
LSTM_test_outputs = np.array(LSTM_test_outputs)


# return the length of data
def _length(data):
    used = tf.sign(tf.reduce_max(tf.abs(data), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length


#Editing
learning_rate = 0.05
n_input = int(4)
num_hidden = 128
n_classes = 2
KEEP_PROB = 1.0


original_graph = tf.Graph()
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
sess_original = tf.Session(graph = original_graph)
with original_graph.as_default():
    # ***************
    # Set placeholder
    # ***************
        data_o = tf.placeholder(tf.float32, [None,window_len-1,n_input])
        target_o = tf.placeholder(tf.float32, [None,n_classes])
        keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
    # ******************
    # Recurrent network.
    # ******************
        with tf.variable_scope('RNN_network'):
            output_o, last_o = rnn.dynamic_rnn(
                rnn_cell.GRUCell(num_hidden),
                data_o,
                dtype=tf.float32,
                sequence_length=_length(data_o),
                time_major=False
            )
        
        last_o = tf.nn.dropout(last_o, keep_prob)

        in_size_o = num_hidden
        out_size_o = int(target_o.get_shape()[1])
        weight_o = tf.truncated_normal([in_size_o, out_size_o], stddev=0.01)
        bias_o = tf.constant(0.1, shape=[out_size_o])
        weight_o = tf.Variable(weight_o)
        bias_o = tf.Variable(bias_o)

        prediction_o = tf.matmul(last_o, weight_o) + bias_o

    # *****************************************
    # Setting Loss and Optimizer & Initializing
    # *****************************************
        loss_o = tf.reduce_mean(  tf.nn.softmax_cross_entropy_with_logits(labels=target_o, logits=prediction_o)  )
        optimizer_o = tf.train.AdamOptimizer(learning_rate).minimize(loss_o)
        

        init_o = tf.global_variables_initializer()
        saver_o = tf.train.Saver()  

tf.reset_default_graph()


if __name__ == '__main__':
    sess_original.run(init_o)
    print('------Training------')
    count = 0
    itr_count = 0
    prev_error = 0.0
    for epoch in range(int(csvCursor_r_num*0.75)):
        for _ in range(5):
            count = count + 1
            sess_original.run([optimizer_o],{data_o: LSTM_training_inputs, target_o: LSTM_training_outputs, keep_prob:0.3})

        #print epoch,count
        try:
            error = sess_original.run(loss_o, {data_o: LSTM_training_inputs, target_o: LSTM_training_outputs, keep_prob:0.3})
            
        except Exception as e:
            print("Error : {0}".format(str(e.args[0])).encode("utf-8"))

        
        if (error - prev_error) < 0.00001:   
            itr_count = itr_count + 1
        if itr_count > 30:
            break
        prev_error = error

        if ( (epoch+1) % 10) == 0:
                print('Epoch {:2d} error {:3.10f} '.format(epoch + 1, error))


if __name__ == '__main__':   
    accuracy = 0
    for i in range(0,100):    
        outputs_o = sess_original.run(prediction_o, {data_o :LSTM_test_inputs, keep_prob:1.})

        acc = 0
        counter = 0
        for t in outputs_o:
            if np.argmax(t) == np.argmax(LSTM_test_outputs[counter]):
                acc = acc + 1


            counter = counter + 1  
        accuracy = accuracy + (float(acc) / counter)
    print('accuracy ->'+str(accuracy/100))

