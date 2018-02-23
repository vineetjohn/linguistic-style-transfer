
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
from tensorflow.python.client import device_lib


# In[ ]:


def get_available_gpus():
    """ Get available GPU devices info. """
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

print(get_available_gpus())


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


# # Google news pretrained vectors
# wv_model_path = word_vector_path + "GoogleNews-vectors-negative300.bin.gz"
# wv_model_1 = gensim.models.KeyedVectors.load_word2vec_format(
#     wv_model_path, binary=True, unicode_errors='ignore')


# In[ ]:


def get_word2vec_embedding(word, model, dimensions):

    vec_rep = np.zeros(dimensions)
    if word in model:
        vec_rep = model[word]
    
    return vec_rep


# In[ ]:


encoder_embedding_matrix = np.random.rand(VOCAB_SIZE + 1, EMBEDDING_SIZE).astype('float32')
decoder_embedding_matrix = np.random.rand(VOCAB_SIZE + 1, EMBEDDING_SIZE).astype('float32')


# In[ ]:


# i = 0
# for key in text_tokenizer.word_index:
#     encoder_embedding_matrix[i] = get_word2vec_embedding(key, wv_model_1, 300)
#     decoder_embedding_matrix[i] = get_word2vec_embedding(key, wv_model_1, 300)
#     i += 1
#     if i >= VOCAB_SIZE:
#         break    


# In[ ]:


print(encoder_embedding_matrix.dtype, decoder_embedding_matrix.dtype)
print(encoder_embedding_matrix.shape, decoder_embedding_matrix.shape)


# ---

# ## Deep Learning Model

# ### Setup Instructions

# In[ ]:


class GenerativeAdversarialNetwork():

    def __init__(self):
        self.style_embedding_size = 128
        self.content_embedding_size = 128
        self.start_token = tf.constant(SOS_INDEX)
        self.end_token = tf.constant(EOS_INDEX)
    
    def get_sentence_representation(self, embedded_sequence):

        lstm_cell_fw = tf.contrib.rnn.DropoutWrapper(
            cell=tf.contrib.rnn.LSTMCell(num_units=128),
            input_keep_prob=0.75,
            output_keep_prob=0.75,
            state_keep_prob=0.75
        )

        lstm_cell_bw = tf.contrib.rnn.DropoutWrapper(
            cell=tf.contrib.rnn.LSTMCell(num_units=128),
            input_keep_prob=0.75,
            output_keep_prob=0.75,
            state_keep_prob=0.75
        )

        _, rnn_states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=lstm_cell_fw, cell_bw=lstm_cell_bw, 
            inputs=embedded_sequence, 
            sequence_length=self.sequence_lengths,
            dtype=tf.float32, time_major=False)

        sentence_representation_dense = tf.concat(
            values=[rnn_states[0].h, rnn_states[1].h], axis=1)

        sentence_representation = tf.nn.dropout(
            x=sentence_representation_dense, keep_prob=0.75, 
            name="sentence_representation")

        return sentence_representation

    def get_content_representation(self, sentence_representation):
        
        content_representation_dense = tf.layers.dense(
            inputs=sentence_representation, units=self.content_embedding_size, 
            activation=tf.nn.relu)
        
        content_representation = tf.nn.dropout(
            x=content_representation_dense, keep_prob=0.75, 
            name="content_representation")

        return content_representation

    def get_style_representation(self, sentence_representation):
        
        style_representation_dense = tf.layers.dense(
            inputs=sentence_representation, units=self.style_embedding_size, 
            activation=tf.nn.relu)
        
        style_representation = tf.nn.dropout(
            x=style_representation_dense, keep_prob=0.75, 
            name="style_representation")
            
        return style_representation

    def get_label_prediction(self, content_representation):

        dense_1 = tf.layers.dense(
            inputs=content_representation, units=NUM_LABELS, 
            activation=tf.nn.relu)
        
        softmax_output = tf.nn.softmax(dense_1, name="label_prediction")

        return softmax_output
    
    
    def generate_output_sequence(self, embedded_sequence, style_representation, 
                                 content_representation, decoder_embeddings):
        
        generative_embedding_dense = tf.concat(
            values=[style_representation, content_representation], axis=1)
        generative_embedding = tf.nn.dropout(
            x=generative_embedding_dense, keep_prob=0.75,
            name="generative_embedding")
        print("generative_embedding: {}".format(generative_embedding))
        
        decoder_cell = tf.contrib.rnn.DropoutWrapper(
            cell=tf.nn.rnn_cell.LSTMCell(
                num_units=128, state_is_tuple=False),
            input_keep_prob=0.75,
            output_keep_prob=0.75,
            state_keep_prob=0.75
        )
        
        batch_sequence_lengths = tf.scalar_mul(
            scalar=MAX_SEQUENCE_LENGTH, 
            x=tf.ones([self.batch_size], dtype=tf.int32))
#         print("batch_sequence_lengths: {}".format(batch_sequence_lengths))
        
        sos_tokens = tf.scalar_mul(
            scalar=self.start_token, 
            x=tf.ones([self.batch_size], dtype=tf.int32))
#         print("sos_tokens: {}".format(sos_tokens))
        
        def get_training_decoder_output():
            training_helper = tf.contrib.seq2seq.TrainingHelper(
                inputs=embedded_sequence, 
                sequence_length=batch_sequence_lengths)
            
            training_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=decoder_cell, helper=training_helper, 
                initial_state=generative_embedding,
                output_layer=tf.layers.Dense(
                    units=VOCAB_SIZE, activation=tf.nn.relu))

            # Dynamic decoding
            training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=training_decoder, impute_finished=True,
                maximum_iterations=MAX_SEQUENCE_LENGTH)
            
            return training_decoder_output

        def get_inference_decoder_output():  
            greedy_embedding_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                embedding=decoder_embeddings, 
                start_tokens=sos_tokens, 
                end_token=self.end_token)

            inference_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=decoder_cell, helper=greedy_embedding_helper, 
                initial_state=generative_embedding,
                output_layer=tf.layers.Dense(
                    units=VOCAB_SIZE, activation=tf.nn.relu))

            # Dynamic decoding
            inference_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=inference_decoder, impute_finished=True, 
                maximum_iterations=MAX_SEQUENCE_LENGTH)

            return inference_decoder_output
        
        final_decoder_output = tf.cond(
            pred=self.training_phase, 
            true_fn=get_training_decoder_output, 
            false_fn=get_inference_decoder_output,
            name="training_inference_conditional")

        sequence_prediction = tf.nn.softmax(
            logits=final_decoder_output.rnn_output,
            name="sequence_prediction")

        return sequence_prediction


    def build_model(self):
        
        with tf.name_scope("input_placeholders"):
            self.input_sequence = tf.placeholder(
                dtype=tf.int32, shape=[None, MAX_SEQUENCE_LENGTH], 
                name="input_sequence")
            print("input_sequence: {}".format(self.input_sequence))

            self.input_label = tf.placeholder(
                dtype=tf.float32, shape=[None, NUM_LABELS], 
                name="input_label")
            print("input_label: {}".format(self.input_label))

            self.sequence_lengths = tf.placeholder(
                dtype=tf.int32, shape=[None], 
                name="sequence_lengths")
            print("sequence_lengths: {}".format(self.sequence_lengths))

            self.training_phase = tf.placeholder(
                dtype=tf.bool, name="training_phase")
            print("training_phase: {}".format(self.training_phase))
        
        self.batch_size = tf.shape(self.input_sequence)[0]

        # word embeddings matrix
        with tf.name_scope("encoder_variables"):
            encoder_embeddings = tf.get_variable(
                initializer=encoder_embedding_matrix, dtype=tf.float32, 
                name="encoder_embeddings")
            print("encoder_embeddings: {}".format(encoder_embeddings))
        
            embedded_sequence = tf.nn.embedding_lookup(
                params=encoder_embeddings, ids=self.input_sequence, 
                name="embedded_sequence")
            print("embedded_sequence: {}".format(embedded_sequence))

            # get sentence representation
            sentence_representation = self.get_sentence_representation(
                embedded_sequence)
            print("sentence_representation: {}".format(sentence_representation))

            # get content representation
            content_representation = self.get_content_representation(
                sentence_representation)
            print("content_representation: {}".format(content_representation))

            # get style representation
            style_representation = self.get_style_representation(
                sentence_representation)
            print("style_representation: {}".format(style_representation))

        # use content representation to predict a label
        with tf.name_scope("adversarial_discriminator_variables"):
            self.label_prediction = self.get_label_prediction(
                content_representation)
            print("label_prediction: {}".format(self.label_prediction))

            self.adversarial_loss = tf.losses.softmax_cross_entropy(
                onehot_labels=self.input_label, logits=self.label_prediction)
            print("adversarial_loss: {}".format(self.adversarial_loss))
        
        # generate new sentence
        with tf.name_scope("decoder_variables"):
            decoder_embeddings = tf.get_variable(
                initializer=decoder_embedding_matrix, dtype=tf.float32, 
                name="decoder_embeddings")
            print("decoder_embeddings: {}".format(decoder_embeddings))

            self.generated_logits = self.generate_output_sequence(
                embedded_sequence, style_representation, content_representation,
                decoder_embeddings)
            print("generated_logits: {}".format(self.generated_logits))

            output_sequence_mask = tf.sequence_mask(
                lengths=self.sequence_lengths, maxlen=MAX_SEQUENCE_LENGTH, 
                dtype=tf.float32)
            print("output_sequence_mask: {}".format(output_sequence_mask))

            self.reconstruction_loss = tf.contrib.seq2seq.sequence_loss(
                logits=self.generated_logits, targets=self.input_sequence, 
                weights=output_sequence_mask)
            print("reconstruction_loss: {}".format(self.reconstruction_loss))
        
        # loss summaries for tensorboard logging
        self.adversarial_loss_summary = tf.summary.scalar(
            tensor=self.adversarial_loss, name="adversarial_loss")

        self.reconstruction_loss_summary = tf.summary.scalar(
            tensor=self.reconstruction_loss, name="reconstruction_loss")
        
    def get_batch_indices(self, batch_size, batch_number, data_limit):
        
        start_index = batch_number * batch_size
        end_index = (batch_number + 1) * batch_size

        end_index = data_limit if end_index > data_limit else end_index

        return (start_index, end_index)


    def run_batch(self, start_index, end_index, training_phase, fetches):
        
        ops = sess.run(
            fetches=fetches, 
            feed_dict={
                self.input_sequence: padded_sequences[
                    start_index : end_index],
                self.input_label: one_hot_labels[
                    start_index : end_index],
                self.sequence_lengths: text_sequence_lengths[
                    start_index : end_index],
                self.training_phase: training_phase
            })
        
        return ops
        
    def train(self, sess):

        writer = tf.summary.FileWriter(
            logdir="/tmp/tensorflow_logs/" + dt.now().strftime("%Y%m%d-%H%M%S") + "/", 
            graph=sess.graph)
        
        adversarial_training_optimizer = tf.train.AdamOptimizer()
        adversarial_training_operation = adversarial_training_optimizer.minimize(
            self.adversarial_loss)
        
        reconstruction_training_optimizer = tf.train.AdamOptimizer()
        reconstruction_training_operation = reconstruction_training_optimizer.minimize(
            self.reconstruction_loss - self.adversarial_loss)
        
        sess.run(tf.global_variables_initializer())

        epoch_reporting_interval = 1
        self.training_examples_size = DATA_SIZE
        training_epochs = 3
        batch_size = 100
        num_batches = self.training_examples_size // batch_size
        print("Training - texts shape: {}; labels shape {}"
              .format(padded_sequences[:self.training_examples_size].shape, 
                      one_hot_labels[:self.training_examples_size].shape))

        for current_epoch in range(1, training_epochs + 1):
            for batch_number in range(num_batches + 1):
                
                (start_index, end_index) = self.get_batch_indices(
                    batch_size=batch_size, batch_number=batch_number, 
                    data_limit=DATA_SIZE)
                
                if start_index == end_index:
                    break
                
                fetches = [adversarial_training_operation, 
                           self.adversarial_loss, 
                           self.adversarial_loss_summary, 
                           reconstruction_training_operation, 
                           self.reconstruction_loss, 
                           self.reconstruction_loss_summary]
                
                _, adv_loss, adv_loss_sum, _, rec_loss, rec_loss_sum = self.run_batch(
                    start_index=start_index, end_index=end_index, training_phase=True, 
                    fetches=fetches)
                    
            writer.add_summary(adv_loss_sum, current_epoch)
            writer.add_summary(rec_loss_sum, current_epoch)
            writer.flush()

            if (current_epoch % epoch_reporting_interval == 0):
                print("Training epoch: {}; Reconstruction loss: {}; Adversarial loss {}"                      .format(current_epoch, rec_loss, adv_loss))
        
        writer.close()

    def infer(self, sess):
        
        generated_sequences = list()
        
        samples_size = 100
        batch_size = 100
        num_batches = samples_size // batch_size
        
        for batch_number in range(num_batches + 1):

            (start_index, end_index) = self.get_batch_indices(
                batch_size=batch_size, batch_number=batch_number, 
                data_limit=samples_size)

            if start_index == end_index:
                break
            
            generated_sequences_batch = self.run_batch(
                start_index=start_index, end_index=end_index,
                training_phase=False, fetches=self.generated_logits)
            
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


# In[ ]:


generated_sequences = gan.infer(sess)


# In[ ]:


index_word_inverse_map = {v: k for k, v in text_tokenizer.word_index.items()}

def generate_word(word_embedding):
    return np.argmax(word_embedding)

def generate_sentence(floating_index_sequence):
    word_indices = map(generate_word, floating_index_sequence)
    word_indices = list(filter(lambda x: x > 0, word_indices))
#     print(word_indices)
    
    words = list(map(lambda x: index_word_inverse_map[x], word_indices))
#     print(words)
    
    return words


# In[ ]:


word_lists = list(map(generate_sentence, generated_sequences))

for word_list in word_lists:
    print(len(list(word_list)))
    print(" ".join(word_list))
# print(list(map(lambda x: len(list(x)), word_lists)))
# print(list(map(lambda x: " ".join(x), word_lists)))


# In[ ]:


# output_file_path = "output/generated_sentences_{}.txt".format(dt.now().strftime("%Y%m%d-%H%M%S"))
# with open(output_file_path, 'w') as output_file:
#     for disjoint_sentence in word_lists:
#         output_file.write(" ".join(disjoint_sentence) + "\n")

