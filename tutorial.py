from autoencoder import AutoEncoder, DataIterator

# train data
datas = [
            [1,1,1,0,0,0],
            [0,0,0,1,1,1]
        ]

# data wrapper
iterator = DataIterator(datas)

# train autoencoder
# assume the input dimension is input_d
# the network is like input_d -> 4 -> 2 -> 4 -> input_d
autoencoder = AutoEncoder()
autoencoder.fit([4, 2], iterator, learning_rate = 0.01, max_epoch = 5000)

# after training

# encode data
encoded_datas = autoencoder.encode(datas)
print "encoder ================"
print encoded_datas 

# decode data
decoded_datas = autoencoder.decode(encoded_datas)
print "decoder ================"
print decoded_datas

# reconstruct data (encode and decode data)
reconstructed_datas = autoencoder.reconstruct(datas)
print "reconstruct ================"
print reconstructed_datas

autoencoder.close()
