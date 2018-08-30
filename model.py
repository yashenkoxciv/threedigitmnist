import numpy as np
from keras.models import Model
from keras.layers import Embedding, Conv2D, LSTM, Dense, Activation
from keras.layers import TimeDistributed, Lambda, BatchNormalization
from keras.layers import Input, RepeatVector, Dropout, Concatenate, Flatten
from keras.datasets import mnist
from threedigitmnist import NumbersVocabulary

# constatnts
Z_DIM = 100
BATCH_SIZE = 30
EPOCHS = 10
BATCHES = 100 #60000 / BATCH_SIZE

# data

nv = NumbersVocabulary(3, '<beg>', '<end>', '<unk>', '<abs>')
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# helper(s)

def conv_layer(x, kernels):
    conv = Conv2D(kernels, kernel_size=3, strides=2, padding='same')(x)
    bn = BatchNormalization()(conv)
    act = Activation('relu')(bn)
    return act

# model

image_input = Input([28, 28*3, 1], name='image_input')

ih1 = conv_layer(image_input, 30)
ih2 = conv_layer(ih1, 60)
ih3 = conv_layer(ih2, Z_DIM)
flat_h = Flatten()(ih3)
image_z = Dense(Z_DIM)(flat_h)

image_encoder = Model(image_input, image_z, name='image_encoder')


caption_input = Input([None], name='caption_input')

embeddings = Embedding(len(nv), Z_DIM)

caption_e = embeddings(caption_input)

caption_encoder = Model(caption_input, caption_e, name='caption_encoder')


m = Concatenate(1)([
        RepeatVector(1)(image_encoder(image_input)),
        caption_encoder(caption_input)
])
hs = LSTM(100, return_sequences=True)(m)
hs = Lambda(lambda x: x[:, 1:, :])(hs)
hyp_c = TimeDistributed(Dense(len(nv), activation='softmax'))(hs)

model = Model([image_input, caption_input], hyp_c, name='caption_model')
model.compile('adam', 'categorical_crossentropy')
#model.summary()

def batch_generator_v1(x, y, batch_size):
    while True:
        yield nv.batch(x, y, batch_size)

# excludes numbers starting with 'three ...'
def batch_generator_v2(x, y, batch_size):
    while True:
        bx, by = nv.batch(x, y, batch_size)
        not_three_hundred_idx = np.argwhere(
                bx[1][:, 1] != nv.v['three']
        ).flatten()
        yield [
                bx[0][not_three_hundred_idx],
                bx[1][not_three_hundred_idx]], by[not_three_hundred_idx]

model.fit_generator(
        batch_generator_v2(x_train, y_train, BATCH_SIZE),
        validation_data=nv.batch(x_test, y_test, 1000),
        steps_per_epoch=BATCHES, epochs=EPOCHS, verbose=1
)

model.save('3dm.pkl')

