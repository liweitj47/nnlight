import sys
sys.path.append("../src")
print sys.path
import nnlight
import numpy

nn_module = {
    "param" : [
        {"name":"kernel_depth", "value":40},
        {"name":"kernel_width", "value":3},
        {"name":"wordvec_size", "value":200},
        {"name":"l2-norm", "value":0.01},

    ],

    "input" : [
        {"name":"label", "dim":1},
        {"name":"words", "dim":3},
        {"name":"mask", "dim":2}
    ],

    "weight" : [
        {"name":"W_conv", "size": ["kernel_depth", 1, "kernel_width", "wordvec_size"] },
        {"name":"W_output", "size":["kernel_depth"] }
    ],

    "layer" : [
        {
            "name":"conv_layer",
            "type":"sentence_convolution",
            "input" : ["words", "mask", "W_conv"]
        },

        {
            "name":"pool_layer",
            "type":"max_pooling",
            "input" : ["conv_layer"]
        },


        {
            "name":"output_layer",
            "type":"logistic",
            "input": ["pool_layer", "W_output"]
        }
    ],

    "loss" : [
        {
            "name":"logliklihood",
            "type":"binary_cross_entropy",
            "input": ["output_layer", "label"]
        }

    ],


    "training" : {
        "loss": "logliklihood",
        "train_info": ["logliklihood", "logliklihood.predict"],
        "test_info": ["logliklihood"]
    }
}

data_dict = {
    "label" : numpy.load(open("sentiment/data/train_labels.npy","rb")),
    "words" : numpy.load(open("sentiment/data/train_words.npy","rb")),
    "mask" : numpy.load(open("sentiment/data/train_mask.npy","rb"))
}
print [(n, v.shape) for n,v in data_dict.items()]
network = nnlight.create(nn_module, data_dict)
a, b = network.train(0,100)
print b
