import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models

from util import ID_TO_CLASS


class MyBasicAttentiveBiGRU(models.Model):

    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int = 128, training: bool = False):
        super(MyBasicAttentiveBiGRU, self).__init__()

        self.num_classes = len(ID_TO_CLASS)

        self.decoder = layers.Dense(units=self.num_classes)
        self.omegas = tf.Variable(tf.random.normal((hidden_size*2, 1)))
        self.embeddings = tf.Variable(tf.random.normal((vocab_size, embed_dim)))

        ### TODO(Students) START
        # ...
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.fw_layer = layers.GRU(self.hidden_size, return_sequences = True)

        # bw_layer = layers.GRUCell(self.hidden_size, return_sequnces = True, go_backwards = True)
        self.bi_layer = layers.Bidirectional(self.fw_layer, input_shape = (self.vocab_size, 2*self.embed_dim), dtype = tf.float32)

        self.training = training
        ### TODO(Students) END

    def attn(self, rnn_outputs):
        ### TODO(Students) START
        # ...
        m_inputs = tf.tanh(rnn_outputs)
        # print(self.omegas.get_shape())
        # exit(0)
        dp_transpose = tf.tensordot(self.omegas, m_inputs, axes = (0, 2))
        dp = tf.transpose(dp_transpose, [1,2,0])
        alpha = tf.nn.softmax(dp)

        weighted_output = tf.reduce_sum(tf.multiply(alpha, rnn_outputs), axis = 1)

        output = tf.tanh(weighted_output)
        # print(output.get_shape)
        # exit(0)
        ### TODO(Students) END

        return output


    def call(self, inputs, pos_inputs, text_tokens, training):
        tokens_mask = tf.cast(inputs!=0, tf.float32)

        word_embed = tf.nn.embedding_lookup(self.embeddings, inputs)
        pos_embed = tf.nn.embedding_lookup(self.embeddings, pos_inputs)

        ### TODO(Students) START
        # ...
        embeds = tf.concat([word_embed, pos_embed], 2)
        # embeds = word_embed
        rnn_output = self.bi_layer(embeds, mask = tokens_mask)
        attn_output = self.attn(rnn_output)
        logits = self.decoder(attn_output)
        
        ### TODO(Students) END

        return {'logits': logits}


class MyAdvancedModel(models.Model):

    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int = 128, training: bool = False):
        super(MyAdvancedModel, self).__init__()
        ### TODO(Students) START
        # ...

        self.num_classes = len(ID_TO_CLASS)

        self.decoder = layers.Dense(units=self.num_classes, activation = "softmax")
        self.omegas = tf.Variable(tf.random.normal((hidden_size*2, 1)))
        self.embeddings = tf.Variable(tf.random.normal((vocab_size, embed_dim)))

        ### TODO(Students) START
        # ...
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
               
        self.conv_layer = layers.Conv1D(filters = 128, kernel_size = 3, 
                                        padding = "same", strides = 1, activation = "relu",
                                        data_format="channels_last", input_shape = (None, None, 1))

        self.conv2d_layer = layers.Conv2D(filters = 64, kernel_size = 3, padding = "same", 
                            activation = "relu", data_format="channels_last", input_shape = (None, None, 1))

        self.maxpool_layer = layers.GlobalMaxPool1D() 
        self.maxpool2d_layer = layers.GlobalMaxPool2D() 

        self.flatten_layer = layers.Flatten()
        self.training = training

        self.dropout = layers.Dropout(0.4)

        ### TODO(Students END


    def call(self, inputs, pos_inputs, text_tokens, training):


        # raise NotImplementedError
        ### TODO(Students) START
        # ...
        # tokens_mask = tf.cast(inputs!=0, tf.float32)

        word_embed = tf.nn.embedding_lookup(self.embeddings, inputs)
        pos_embed = tf.nn.embedding_lookup(self.embeddings, pos_inputs)

        e1_index = []
        e2_index = []
        p1_index = []
        p2_index = []

        ## Calculating the distance matrices of each word in the sentences provided  from the two entities
        ## Additionally I passed text tokens as an additional input to calculate the distance matrices as mentioned in the paper

        for w,p,list_token in zip(inputs,pos_inputs,text_tokens):
            w = np.array(w).tolist()
            p = np.array(p).tolist()
            
            count = 0
            entity_index = []
            none_index = []

            list_token = list_token.tolist()
            for token in list_token:
                if token is not None and "e11" in token and "e12" in token:
                    entity_index.append(count)
                if token is not None and "e21" in token and "e22" in token:
                    entity_index.append(count)
                if token is None:
                    none_index.append(count)

                count = count + 1

            entity1 = w[entity_index[0]]
            entity2 = w[entity_index[1]]
            pentity1 = p[entity_index[0]]
            pentity2 = p[entity_index[1]]


            rep1 = [entity1] * len(w)
            rep2 = [entity2] * len(w)
            prep1 = [pentity1] * len(w)
            prep2 = [pentity2] * len(w)

            if len(none_index) > 0:
                for ind in none_index:  
                    rep1[ind] = 0
                    rep2[ind] = 0
                    prep1[ind] = 0
                    prep2[ind] = 0


            e1_index.append(rep1)
            e2_index.append(rep2)
            p1_index.append(prep1)
            p2_index.append(prep2)



 
        e1_index = np.array(e1_index)
        e2_index = np.array(e2_index)
        p1_index = np.array(p1_index)
        p2_index = np.array(p2_index)


        ent1_embed = tf.nn.embedding_lookup(self.embeddings, e1_index)
        ent2_embed = tf.nn.embedding_lookup(self.embeddings, e2_index)

        pent1_embed = tf.nn.embedding_lookup(self.embeddings, p1_index)
        pent2_embed = tf.nn.embedding_lookup(self.embeddings, p2_index)

        dist1_embed = tf.math.subtract(word_embed, ent1_embed)
        dist2_embed = tf.math.subtract(word_embed, ent1_embed)

        pdist1_embed = tf.math.subtract(pos_embed, pent1_embed)
        pdist2_embed = tf.math.subtract(pos_embed, pent1_embed)

        embeds = tf.concat([word_embed, dist1_embed, dist2_embed], 2)
        # embeds = tf.concat([word_embed, pos_embed], 2)
        embeds2d = tf.expand_dims(embeds, 3)




        # cnn_output = self.conv_layer(embeds)
        # maxpool_output = self.maxpool_layer(cnn_output)

        cnn_output = self.conv2d_layer(embeds2d)
        maxpool_output = self.maxpool2d_layer(cnn_output)

        dropout_output = self.dropout(maxpool_output)
        logits = self.decoder(dropout_output)

        ### TODO(Students END

        return {'logits': logits}

