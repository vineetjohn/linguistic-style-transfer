
# coding: utf-8

# # Authorship Style Transfer

# In[ ]:


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


# In[ ]:


# !/usr/local/cuda-8.0/extras/demo_suite/deviceQuery
# !nvidia-smi


# In[ ]:


import numpy as np
import tensorflow as tf
import gensim

from datetime import datetime as dt
from IPython.display import display, HTML
from tensorflow.python.client import device_lib


# In[ ]:


def get_available_gpus():
    """ Get available GPU devices info. """
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

print(get_available_gpus())


# In[ ]:


def browser_alert(message):
    display(HTML(
        '<script type="text/javascript">alert("' + message + '");</script>'))
    
def browser_notify(message):
    display(HTML(
        '<script type="text/javascript">var notification=new Notification("\
         Jupyter Notification",{icon:"http://blog.jupyter.org/content/images\
         /2015/02/jupyter-sq-text.png",body:"' + message + '"});</script>'))


# ---

# ## Data Preprocessing

# In[ ]:


text_file_path = "data/c50-articles-dev.txt"
label_file_path = "data/c50-labels-dev.txt"


# ### Conversion of texts into integer sequences

# In[ ]:


VOCAB_SIZE = 1000
EMBEDDING_SIZE = 300


# In[ ]:


text_tokenizer = tf.keras.preprocessing.text.Tokenizer(
    num_words=VOCAB_SIZE, filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')

with open(text_file_path) as text_file:
    text_tokenizer.fit_on_texts(text_file)
    
with open(text_file_path) as text_file:
    integer_text_sequences = text_tokenizer.texts_to_sequences(text_file)

text_sequence_lengths = np.asarray(
    a=list(map(lambda x: len(x), integer_text_sequences)), dtype=np.int32)

MAX_SEQUENCE_LENGTH = np.amax(text_sequence_lengths)

padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
     integer_text_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

print("text_sequence_lengths: {}".format(text_sequence_lengths.shape))
print("padded_sequences.shape: {}".format(padded_sequences.shape))


# In[ ]:


print("MAX_SEQUENCE_LENGTH: {}".format(MAX_SEQUENCE_LENGTH))


# In[ ]:


SOS_INDEX = text_tokenizer.word_index['<sos>']
EOS_INDEX = text_tokenizer.word_index['<eos>']
DATA_SIZE = padded_sequences.shape[0]


# In[ ]:


# text_tokenizer.word_index


# ### Conversion of labels to one-hot represenations

# In[ ]:


label_tokenizer =  tf.keras.preprocessing.text.Tokenizer(lower=False)

with open(label_file_path) as label_file:
    label_tokenizer.fit_on_texts(label_file)

with open(label_file_path) as label_file:
    label_sequences = label_tokenizer.texts_to_sequences(label_file)

NUM_LABELS = len(label_tokenizer.word_index)
one_hot_labels = np.asarray(list(
    map(lambda x: np.eye(NUM_LABELS, k=x[0])[0], label_sequences)))

print("one_hot_labels.shape: {}".format(one_hot_labels.shape))


# ### Initializing Pre-trained Embeddings

# In[ ]:


word_vector_path = "word-embeddings/"


# In[ ]:


# Google news pretrained vectors
wv_model_path = word_vector_path + "GoogleNews-vectors-negative300.bin.gz"
wv_model_1 = gensim.models.KeyedVectors.load_word2vec_format(
    wv_model_path, binary=True, unicode_errors='ignore')


# In[ ]:


def get_word2vec_embedding(word, model, dimensions):

    vec_rep = np.zeros(dimensions)
    if word in model:
        vec_rep = model[word]
    
    return vec_rep


# In[ ]:


pretrained_embedding_matrix = np.zeros((VOCAB_SIZE, EMBEDDING_SIZE), dtype=float)


# In[ ]:


i = 0
for key in text_tokenizer.word_index:
    pretrained_embedding_matrix[i] = get_word2vec_embedding(key, wv_model_1, 300)
    i += 1
    if i >= VOCAB_SIZE:
        break
    


# In[ ]:


pretrained_embedding_matrix.shape


# ---

# ## Deep Learning Model

# ### Setup Instructions

# In[ ]:


class GenerativeAdversarialNetwork():

    def __init__(self):
        self.style_embedding_size = 128
        self.content_embedding_size = 128
        self.batch_size = 100
    
    def get_sentence_representation(self, embedded_sequence):

        lstm_cell_fw = tf.contrib.rnn.LSTMCell(
            num_units=128)
        lstm_cell_bw = tf.contrib.rnn.LSTMCell(
            num_units=128)

        _, rnn_states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=lstm_cell_fw, cell_bw=lstm_cell_bw, 
            inputs=embedded_sequence, 
            dtype=tf.float32, time_major=False)
        sentence_representation = tf.concat(
            values=[rnn_states[0].h, rnn_states[1].h], axis=1, 
            name="sentence_representation")

        return sentence_representation

    def get_content_representation(self, sentence_representation):
        
        dense_content = tf.layers.dense(
            inputs=sentence_representation, units=self.content_embedding_size, 
            activation=tf.nn.relu, name="content_representation")

        return dense_content

    def get_style_representation(self, sentence_representation):
        
        dense_style = tf.layers.dense(
            inputs=sentence_representation, units=self.style_embedding_size, 
            activation=tf.nn.relu, name="style_representation")
        return dense_style

    def get_label_prediction(self, content_representation):

        dense_1 = tf.layers.dense(
            inputs=content_representation, units=NUM_LABELS, 
            activation=tf.nn.relu, name="dense_1")
        
        softmax_output = tf.nn.softmax(dense_1, name="label_prediction")

        return softmax_output
    
    def generate_output_sequence(self, embedded_sequence, style_representation, 
                                 content_representation):
        
        generative_embedding = tf.concat(
            values=[style_representation, content_representation], axis=1)
        print("generative_embedding: {}".format(generative_embedding))
        
        decoder_cell = tf.nn.rnn_cell.LSTMCell(
            num_units=128, state_is_tuple=False)
        
        batch_sequence_lengths = tf.scalar_mul(
            scalar=MAX_SEQUENCE_LENGTH, 
            x=tf.ones([self.batch_size], dtype=tf.int32))
        print("batch_sequence_lengths: {}".format(batch_sequence_lengths))
        
        # Helper
        training_helper = tf.contrib.seq2seq.TrainingHelper(
            inputs=embedded_sequence, sequence_length=batch_sequence_lengths)

        # Decoder
        generative_decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=decoder_cell, helper=training_helper, 
            initial_state=generative_embedding,
            output_layer=tf.layers.Dense(
                units=VOCAB_SIZE, activation=tf.nn.relu))
        
        # Dynamic decoding
        final_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder=generative_decoder,
            maximum_iterations=MAX_SEQUENCE_LENGTH)
        print("final_decoder_output: {}".format(final_decoder_output))
        
        softmax_prediction = tf.nn.softmax(logits=final_decoder_output.rnn_output)

        return softmax_prediction


    def build_model(self):
        
        self.input_sequence = tf.placeholder(
            dtype=tf.int32, shape=[self.batch_size, MAX_SEQUENCE_LENGTH], 
            name="input_sequence")
        print("input_sequence: {}".format(self.input_sequence))

        self.input_label = tf.placeholder(
            dtype=tf.float32, shape=[self.batch_size, NUM_LABELS], 
            name="input_label")
        print("input_label: {}".format(self.input_label))

        # word embeddings matrix
        word_embeddings = tf.get_variable(
            shape=[VOCAB_SIZE, EMBEDDING_SIZE], dtype=tf.float32, 
            name="word_embeddings")
        self.embedding_placeholder = tf.placeholder(
            dtype=tf.float32, shape=[VOCAB_SIZE, EMBEDDING_SIZE],
            name="embedding_placeholder")
        self.embedding_init = word_embeddings.assign(self.embedding_placeholder)
        print("word_embeddings: {}".format(word_embeddings))
        
        embedded_sequence = tf.nn.embedding_lookup(
            word_embeddings, self.input_sequence, name="embedded_sequence")
        print("embedded_sequence: {}".format(embedded_sequence))

        # get sentence representation
        sentence_representation = self.get_sentence_representation(
            embedded_sequence)
        print("sentence_representation: {}".format(sentence_representation))

        # get content representation
        content_representation = self.get_content_representation(
            sentence_representation)
        print("content_representation: {}".format(content_representation))

        # use content representation to predict a label
#         self.label_prediction = self.get_label_prediction(
#             content_representation)
#         print("label_prediction: {}".format(self.label_prediction))

#         self.adversarial_loss = tf.losses.softmax_cross_entropy(
#             onehot_labels=self.input_label, logits=self.label_prediction)
#         print("adversarial_loss: {}".format(self.adversarial_loss))

#         self.adversarial_loss_summary = tf.summary.scalar(
#             tensor=self.adversarial_loss, name="adversarial_loss")

        # get style representation
        style_representation = self.get_style_representation(
            sentence_representation)
        print("style_representation: {}".format(style_representation))
        
        # generate new sentence
        self.generated_logits = self.generate_output_sequence(
            embedded_sequence, style_representation, content_representation)
        print("generated_logits: {}".format(self.generated_logits))
        
        self.reconstruction_loss = tf.contrib.seq2seq.sequence_loss(
            logits=self.generated_logits, targets=self.input_sequence, 
            weights=tf.ones(tf.shape(self.input_sequence)))
        print("reconstruction_loss: {}".format(self.reconstruction_loss))

        self.reconstruction_loss_summary = tf.summary.scalar(
            tensor=self.reconstruction_loss, name="reconstruction_loss")


    def train(self, sess):

        writer = tf.summary.FileWriter(
            logdir="/tmp/tensorflow_logs/" + dt.now().strftime("%Y%m%d-%H%M%S") + "/", 
            graph=sess.graph)
        
#         adversarial_training_optimizer = tf.train.AdamOptimizer()
#         adversarial_training_operation = adversarial_training_optimizer.minimize(
#             self.adversarial_loss)
        
        reconstruction_training_optimizer = tf.train.AdamOptimizer()
        reconstruction_training_operation = reconstruction_training_optimizer.minimize(
            self.reconstruction_loss)
        
        sess.run(tf.global_variables_initializer())
        sess.run(
            fetches=self.embedding_init, 
            feed_dict={self.embedding_placeholder: pretrained_embedding_matrix})

        epoch_reporting_interval = 1
        self.training_examples_size = DATA_SIZE
        training_epochs = 100
        num_batches = self.training_examples_size // self.batch_size
        print("Training - texts shape: {}; labels shape {}"
              .format(padded_sequences[:self.training_examples_size].shape, 
                      one_hot_labels[:self.training_examples_size].shape))

        for current_epoch in range(1, training_epochs + 1):
            for batch_number in range(num_batches):
#                 _, adv_loss, adv_loss_sum,
                _, rec_loss, rec_loss_sum = sess.run(
                    fetches=[
#                         adversarial_training_operation, self.adversarial_loss, 
#                         self.adversarial_loss_summary, 
                        reconstruction_training_operation, self.reconstruction_loss, 
                        self.reconstruction_loss_summary], 
                    feed_dict={
                        self.input_sequence: padded_sequences[
                            batch_number * self.batch_size : \
                            (batch_number + 1) * self.batch_size],
                        self.input_label: one_hot_labels[
                            batch_number * self.batch_size : \
                            (batch_number + 1) * self.batch_size]
                    })
#                 writer.add_summary(adv_loss_sum, current_epoch)
            writer.add_summary(rec_loss_sum, current_epoch)
            writer.flush()

            if (current_epoch % epoch_reporting_interval == 0):
                print("Training epoch: {}; Reconstruction Loss: {}"
                      .format(current_epoch, rec_loss))
        
        writer.close()

    def infer(self, sess):
        
        generated_sequences = list()
        
        samples_size = 1000
        num_batches = samples_size // self.batch_size
        
        for batch_number in range(num_batches):
            generated_sequences_batch = sess.run(
                fetches=self.generated_logits, 
                feed_dict={
                    self.input_sequence: padded_sequences[
                        batch_number * self.batch_size : \
                        (batch_number + 1) * self.batch_size],
                    self.input_label: one_hot_labels[
                        batch_number * self.batch_size : \
                        (batch_number + 1) * self.batch_size]
                })
            generated_sequences.extend(generated_sequences_batch)

        return generated_sequences


# ### Train Network

# In[ ]:


tf.reset_default_graph()


# In[ ]:


gan = GenerativeAdversarialNetwork()
gan.build_model()


# In[ ]:


config_proto = tf.ConfigProto()
config_proto.gpu_options.allow_growth=True  
sess = tf.Session(config=config_proto)


# In[ ]:


gan.train(sess)
browser_notify("Training complete")


# In[ ]:


generated_sequences = gan.infer(sess)
browser_notify("Inference complete")


# In[ ]:


index_word_inverse_map = {v: k for k, v in text_tokenizer.word_index.items()}

def generate_word(word_embedding):
    return np.argmax(word_embedding)

def generate_sentence(floating_index_sequence):
    word_indices = map(generate_word, floating_index_sequence)
    word_indices = filter(lambda x: x > 0, word_indices)
    # print(word_indices)
    
    words = map(lambda x: index_word_inverse_map[x], word_indices)
    # print(sentence)
    
    return words


# In[ ]:


word_lists = map(generate_sentence, generated_sequences)
# print(list(map(lambda x: " ".join(x), word_lists)))


# In[ ]:


output_file_path = "data/generated_sentences.txt"
with open(output_file_path, 'w') as output_file:
    for disjoint_sentence in word_lists:
        output_file.write(" ".join(disjoint_sentence) + "\n")


# In[ ]:


browser_notify("Sampling complete")

