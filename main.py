import numpy as np
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
import keras.backend as K
from keras.datasets import mnist


train_size = 640000
validated_size = 128000
L = train_size + validated_size
LL = train_size
N = 8
bit = 16
maxi = 2


def get_data(L=1000, N=8, bit=8, maxi=2):
    data = np.float32(np.random.randint(maxi, size=(L, N*bit)))
    return data

data = get_data(L=L, N=N, bit=bit, maxi=maxi)
train_data = data[:LL]
test_data = data[LL:]


mu = 2
t = int(N*bit/mu)

mapping_data_1 = np.zeros((L, 128))

mapping_table_1 = {
    (0,0) : (-1, -1),
    (0,1) : (-1, +1),
    (1,0) : (+1, -1),
    (1,1) : (+1, +1),
}

demapping_table_1 = {v : k for k, v in mapping_table_1.items()}


for i in range(L):
    bits_reshape = data[i].reshape(-1, 2)
    tmp_map_data = np.array([mapping_table_1[tuple(b)] for b in bits_reshape])
    mapping_data_1[i, :t] = tmp_map_data[:, 0]
    mapping_data_1[i, t:] = tmp_map_data[:, 1]

train_mapping_data1 = mapping_data_1[:LL]
test_mapping_data1 = mapping_data_1[LL:]


mapping_data_2 = np.zeros((L, 128))

mapping_table_2 = { 
    (0,0) : (0, 0), 
    (0,1) : (0, +1),
    (1,0) : (+1, 0),
    (1,1) : (+1, +1),
}

demapping_table_2 = {v : k for k, v in mapping_table_2.items()}


for i in range(L):
    bits_reshape = data[i].reshape(-1, 2)
    tmp_map_data = np.array([mapping_table_2[tuple(b)] for b in bits_reshape])
    mapping_data_2[i, :t] = tmp_map_data[:, 0]
    mapping_data_2[i, t:] = tmp_map_data[:, 1]

train_mapping_data2 = mapping_data_2[:LL]
test_mapping_data2 = mapping_data_2[LL:]


neuros = 2048
encoding_dim = 128
batch_size=32
drop_x = 0.1
nu = 0.001


def IDFT(OFDM_data):
    return K.tf.ifft(OFDM_data)


def get_papr_in_model(data):
    OFDM_value = K.tf.complex(data[:64], data[64:])
    OFDM_samples = IDFT(OFDM_value)
    tmp = OFDM_samples * K.tf.conj(OFDM_samples)
    tmp = K.tf.real(tmp)
    return 10*K.log(K.abs(K.max(tmp)/K.mean(tmp)))/np.log(10)

def paprLossFunction(y_true, y_pred):
    res = 0
    for i in range(batch_size):
        papr_pred = get_papr_in_model(y_pred[i,:])
        papr_true = get_papr_in_model(y_true[i,:])
        papr = (papr_pred - papr_true)**2
        res += papr
    return res/batch_size


def berLossFunction(y_true, y_pred):
    return 0

def build_autoencoder_model(): # 
    input_img = layers.Input(shape=(N*bit,))
    dense_1 = layers.Dense(neuros, activation='relu')(input_img)
    drop = layers.Dropout(drop_x)(dense_1)
    dense_2 = layers.Dense(neuros, activation='relu')(drop)
    drop = layers.Dropout(drop_x)(dense_2)
    encoded = layers.Dense(encoding_dim, activation='relu', name="encoded")(drop)
    
    dense_3 = layers.GaussianNoise(0.2)(encoded) # 加入随机高斯噪声，模拟信道过程，提高模型泛化能力
    
    dense_4 = layers.Dense(neuros, activation='relu')(dense_3)
    drop = layers.Dropout(drop_x)(dense_4)
    dense_5 = layers.Dense(neuros, activation='relu')(drop)
    drop = layers.Dropout(drop_x)(dense_5)
    decoded = layers.Dense(N*bit, activation='sigmoid', name="decoded")(drop)
    
    autoencoder = models.Model(inputs=input_img, outputs=[encoded, decoded])
    encoder = models.Model(input_img, encoded)
    encoded_input = layers.Input(shape=(encoding_dim,))
    deco = autoencoder.layers[-5](encoded_input) # 依层复原 decoder
    deco = autoencoder.layers[-4](deco)
    deco = autoencoder.layers[-3](deco)
    deco = autoencoder.layers[-2](deco)
    deco = autoencoder.layers[-1](deco)
    decoder = models.Model(encoded_input, deco)
    
    sgdWithAdam = optimizers.Adam(lr=0.0001) # 初始化 优化器 
    autoencoder.compile(optimizer=sgdWithAdam, # 设置 优化器
                        loss={'decoded': K.binary_crossentropy, 'encoded': paprLossFunction}, # 分别设置 loss
                        loss_weights={'decoded': 1, 'encoded': nu}, # 设置两个 loss 的系数
                        metrics=[paprLossFunction], # 训练时额外追踪一个loss
                       )
    return autoencoder, encoder, decoder


autoencoder, encoder, decoder = build_autoencoder_model()
autoencoder.summary()


history = autoencoder.fit(train_mapping_data2, [train_mapping_data1, train_mapping_data2],
                          epochs=40,
                          batch_size=batch_size,
                          shuffle=True,
                          validation_data=(test_mapping_data2, [test_mapping_data1, test_mapping_data2]))


def demapping_decoded_data(data):
    res = np.zeros(data.shape)
    tmp = np.zeros((64, 2))
    for i in range(data.shape[0]):
        tmp[:, 0] = data[i][:64] 
        tmp[:, 1] = data[i][64:]
        
        tmp_map_data = np.array([demapping_table_2[tuple(b)] for b in tmp])
        
        res[i, :] = tmp_map_data.reshape((1, -1))
    return res
        
def to_one_hot(data): # 把 sigmoid 的结果转化为几率更大的结果
    shape_buk = data.shape
    data.reshape(shape_buk[0], -1)
    for each in range(data.shape[0]):
        for i in range(data[each].shape[0]):
            if data[each][i] <= 0.5:
                data[each][i] = 0. # 如果是 1 的几率小于 0.5， 则为 0
            else:
                data[each][i] = 1.
    return data.reshape(shape_buk)


encoded_test_data, decoded_test_data = autoencoder.predict(test_mapping_data2)
decoded_dehot_data = to_one_hot(decoded_test_data)

demapping_test_data = demapping_decoded_data(decoded_dehot_data)

error_test = np.absolute(demapping_test_data - test_data)
print("test_data = \n{}".format(test_data[0, :]))
print("decoded_data = \n{}".format(demapping_test_data[0, :]))
print("error = {:.4f} %".format(100*np.sum(error_test)/(test_mapping_data2.shape[0]*test_mapping_data2.shape[1])))


def binary_to_mapping(data):   
    for bits in range(data.shape[0]):
        for bit in range(len(data[bits,:])):
            if data[bits, bit] == 0:
                data[bits, bit] = -1
    return data


def convert_128_to_64(bits):
    return bits[:64] + bits[64:] * 1j


def get_papr(data):
    res = 0
    data = binary_to_mapping(data)
    for bits in range(data.shape[0]):
        mapping_data = convert_128_to_64(data[bits])
        ofdm_samples = np.fft.ifft(mapping_data)
        
        tmp = ofdm_samples * ofdm_samples.conj()
        tmp = np.real(tmp)
        res += 10*np.log10(np.abs(tmp.max()/tmp.mean()))
    return res/data.shape[0]

print("test_mapping_data's papr = {:.4} dB".format(get_papr(test_mapping_data2)))
print("After encoder papr is = {:.4} dB".format(get_papr(encoded_test_data)))


encoded_train_data, decoded_train_data = autoencoder.predict(train_mapping_data2)
decoded_dehot_data = to_one_hot(decoded_train_data)

demapping_train_data = demapping_decoded_data(decoded_dehot_data)

error_train = np.absolute(demapping_train_data - train_data)
print("train_data = \n{}".format(train_data[0, :]))
print("decoded_data = \n{}".format(demapping_train_data[0, :]))
print("error = {:.4f} %".format(100*np.sum(error_train)/(train_mapping_data2.shape[0]*train_mapping_data2.shape[1])))

print("train_mapping_data's papr = {:.4} dB".format(get_papr(train_mapping_data2)))
print("After encoder papr is = {:.4} dB".format(get_papr(encoded_train_data)))

autoencoder.save('autoencoder.h5')
encoder.save('encoder.h5')
decoder.save('decoder.h5')
print("Saved!")
