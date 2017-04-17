# Tensorflow-AutoEncoder
AutoEncoder implemented in Tensorflow

### Requirement

+ python 2.7+
+ Tensorflow >0.10.0 

Some tutorials require:

+ Matplotlib


### Tutorial

```python
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
output_datas = autoencoder.output(datas)
print "reconstruct ================"
print output_datas

autoencoder.close()
```

### Iris Example

[tutorial_iris.py](https://github.com/CrawlScript/Tensorflow-AutoEncoder/blob/master/tutorial_iris.py) encodes iris datasets(3 features) to a 2 features datasets.

<table><tr><td style="width:50%;text-align:center;">Origin Iris Data (3 features)</td><td style="width:50%;text-align:center;">Encoded Iris Data(2 features)</td></tr>
<tr><td><img src="https://raw.githubusercontent.com/CrawlScript/Tensorflow-AutoEncoder/master/tutorial_datasets/iris/imgs/origin_iris_data.png"></img></td><td><img src="https://raw.githubusercontent.com/CrawlScript/Tensorflow-AutoEncoder/master/tutorial_datasets/iris/imgs/encoded_iris_data.png"></img></td></tr>
</table>

