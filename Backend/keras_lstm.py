import jieba as jb
import numpy as np
import keras as krs
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
  
def load_data():
    titles = []
    print("Loading health topic's data......")
    with open("data/health.txt", "r") as f:
        for line in f.readlines():
            titles.append(line.strip())

    print("Loading technology topic's data......")
    with open("data/tech.txt", "r") as f:
        for line in f.readlines():
            titles.append(line.strip())

    print("Loading design topic's data......")
    with open("data/design.txt", "r") as f:
        for line in f.readlines():
            titles.append(line.strip())

    print("A total of %s titles were loaded" % len(titles))

    return titles

def load_label():
    arr0 = np.zeros(shape=[12000, ])
    arr1 = np.ones(shape=[12000, ])
    arr2 = np.array([2]).repeat(7318)
    target = np.hstack([arr0, arr1, arr2])
    print("A total of %s labels were loaded" % target.shape)

    encoder = LabelEncoder()
    encoder.fit(target)
    encoded_target = encoder.transform(target)
    dummy_target = krs.utils.np_utils.to_categorical(encoded_target)

    return dummy_target

titles = load_data()
  
target = load_label()

max_sequence_length = max([len(x.split(" ")) for x in titles])
embedding_size = 50

# Content segmentation
titles = [".".join(jb.cut(t, cut_all=True)) for t in titles]

# Bag of words
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_sequence_length, min_frequency=1)
text_processed = np.array(list(vocab_processor.fit_transform(titles)))

# Words
dict = vocab_processor.vocabulary_._mapping

# Network architecture
def build_netword(num_vocabs):
    model = krs.Sequential()
    model.add(krs.layers.Embedding(num_vocabs, embedding_size, input_length=max_sequence_length))
    model.add(krs.layers.LSTM(32, dropout=0.2, recurrent_dropout=0.2))
    model.add(krs.layers.Dense(3))
    model.add(krs.layers.Activation("softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model
  
num_vocabs = len(dict.items())
model = build_netword(num_vocabs=num_vocabs)

import time
start = time.time()
# Train the model
model.fit(text_processed, target, batch_size=512, epochs=10, )
finish = time.time()
print("Training time: %f seconds" %(finish-start))

# serialize model to JSON
model_json = model.to_json()
with open("keras_lstm.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("keras_lstm.h5")
print("Saved model to disk")