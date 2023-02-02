from keras import utils
from keras.models import load_model
from keras.preprocessing.text import text_to_word_sequence
from keras.datasets import imdb
import numpy as np
from transformers import TFAutoModel, AutoTokenizer, AutoConfig
from glob import glob
import tensorflow as tf

tokenizer = AutoTokenizer.from_pretrained('./model')
config = AutoConfig.from_pretrained('./model', output_hidden_states=True)
config.hidden_dropout_prob = 0
config.attention_probs_dropout_prob = 0
backbone = TFAutoModel.from_pretrained('./model', config=config)

def feedback_model():
    input_ids = tf.keras.layers.Input(shape=(512,), dtype=tf.int32, name="input_ids")
    attention_masks = tf.keras.layers.Input(shape=(512,), dtype=tf.int32, name="attention_masks")
    x = backbone.deberta(input_ids, attention_mask=attention_masks)
    x = x.hidden_states 
    x = tf.stack([MeanPool()(hidden_s, mask=attention_masks) for hidden_s in x[-4:]],axis=2) 
    x = tf.keras.layers.Dense(1, use_bias=False, kernel_constraint=WeightsSumOne())(x)
    x = tf.squeeze(x, axis=-1)
    x = tf.keras.layers.Dense(6, activation="sigmoid")(x)
    x = tf.keras.layers.Rescaling(scale=4.0, offset=1.0)(x)
    model = tf.keras.Model(inputs=[input_ids, attention_masks], outputs=x)
    return model

class MeanPool(tf.keras.layers.Layer):
    def call(self, inputs, mask=None):
  
        broadcast_mask = tf.expand_dims(tf.cast(mask, "float32"), -1)
        embedding_sum = tf.reduce_sum(inputs * broadcast_mask, axis=1)
        mask_sum = tf.reduce_sum(broadcast_mask, axis=1)    
        mask_sum = tf.math.maximum(mask_sum, tf.constant([1e-9]))
        return embedding_sum / mask_sum

class WeightsSumOne(tf.keras.constraints.Constraint):
    def __call__(self, w):
        return tf.nn.softmax(w, axis=0)    
    
def predict(text):
    tokenized = tokenizer(' '.join(text.split('  ')[0:8]), 
                          ' '.join(text.split('  ')[8::]),
                          truncation=True,
                          max_length=512,
                          padding='max_length',
                          return_attention_mask=True,
                          return_tensors="np"
                          )
    ids = []
    masks = []
    ids.append(tokenized['input_ids'][0])
    masks.append(tokenized['attention_mask'][0]) 
    ids = np.array(ids, dtype="int32")
    masks = np.array(masks, dtype="int32")
    model = feedback_model()
    learnings = []
    ds = glob('./*.h5')
    for i in ds:
        model.load_weights(i)
        p = model.predict((ids,masks), batch_size=8)
        learnings.append(p)
    learnings = np.mean(learnings, axis=0)
    return learnings

def process(in_text):
    results = predict(in_text)
        return (round(results[0][0],1), round(results[0][1],1),round(results[0][2],1), round(results[0][3],1), \
                round(results[0][4],1), round(results[0][5],1))
        
