#encoding=utf-8
#author: xinqi.bao
#
import random
import re
import theano
import numpy
from theano import tensor as T

global_config = {}

def get_config():
    cm = {
        NNScalarInt64 : ["int64", "int"],
        NNScalarFloat32 : ["float32", "float"],
        NNArrayFloat32 : ["[float32]", "[float]"],
        NNArrayInt64: ["[int64]", "[int]"],

        SentenceConvLayer: ["sentence_convolution"],
        MaxPoolingLayer: ["max_pooling"],
        SoftmaxLayer : ["softmax"],
        LogisticLayer : ["logistic"],

        BinaryCrossEntropyLoss : ["binary_cross_entropy"]
    }

    constructor_map = {}
    for constructor, dtypes in cm.items():
        for dtype in dtypes:
            constructor_map[dtype] = constructor
    global_config["constructor_map"] = constructor_map

def create(src, data_dict={}):

    if len(global_config)==0:
        get_config()

    if isinstance(src, str):
        return create_from_file(src, data_dict)
    else:
        return create_from_dict(src, data_dict)
        
def create_from_file(config_path, data_dict={}):
    pass

def create_from_dict(d, data_dict={}):
    
    def raise_(s):
        global raise_
        raise_(s)
    
    def get_constructor_(dtype):
        if dtype in ["float","float32"]:
            return NNScalarFloat32
        elif dtype in ["int", "int64"]:
            return NNScalarInt64
        else:
            raise_("dtype '%s' not supported" % dtype)

    def eval_(v):
        if isinstance(v, str):
            refs = v.strip().split(".")
            prefix = []

            node = nn.core["values"].get(refs[0], None)
            if not node:
                raise_("'%s' not found" % refs[0])
            prefix.append(refs[0])
            node = node.node()

            for ref in refs[1:]:
                node = node.node(ref=ref)
                if not node:
                    raise_("'%s' not found in '%s'<%s>" % (ref, str.join(".",prefix), node))
                prefix.append(ref)

            return node

        elif isinstance(v, int):
            return NNScalarInt64(v)
        elif isinstance(v, float):
            return NNScalarFloat32(v)
        elif isinstance(v, NNValue):
            return v
        else:
            raise_("unsupported value '%s'" % v)
   
    def raw_(v):
        if isinstance(v, NNValue):
            return v.value()
        elif isinstance(v, list):
            return map(raw_, v)
        else:
            return v

    def make_input_(dims, dtype):
        if isinstance(dims, int):
            dims = [None for i in range(dims)]
        if isinstance(dims, list):
            if len(dims)==0:
                constructor = config["constructor_map"].get(dtype,None)
                if constructor:
                    return constructor()
            else:
                dtype = "[" + dtype + "]"
                constructor = config["constructor_map"].get(dtype,None)
                if constructor:
                    return constructor(dims=dims)
        raise_("unsupported datatype '%s'" % dtype)
    
    def make_shared_(size, init, dtype):
        basetype = dtype
        dtype = "[" + basetype + "]"
        size = map(eval_, size)

        if not (
            isinstance(size, list) and\
            len(size) > 0 and\
            all([isinstance(x, NNScalarInt) for x in size]) and\
            all([x.is_durable() for x in size])\
          ):
            raise_("size must be an integer array")
        if not basetype in ["float32", "int64"]:
            raise_("data type '%s' not supported for weights" % dtype)
        if not config["constructor_map"].has_key(dtype):
            raise_("data type '%s' not supported for weights" % dtype)
            
        size = map(raw_, size)
        v = numpy.zeros(size, dtype=basetype)
        if init=="random":
            high = 4.0 * numpy.sqrt( 6.0 / sum(size) )
            low = -high
            def __randomize__(w, dims):
                if len(dims)==1:
                    for i in range(dims[0]):
                        w[i] = random.uniform(low,high)
                else:
                    for i in range(dims[0]):
                        __randomize__(w[i], dims[1:])
            __randomize__(v, size)

        constructor = config["constructor_map"][dtype]
        return constructor(dims=size, v=NNData(v))

    def make_layer_(layer):
        name = layer.get("name")
        ltype = layer.get("type", None)
        if not ltype:
            raise_("layer must be explicitly assigned type")

        constructor = config["constructor_map"].get(ltype)
        if not constructor:
            raise_("unsupported layer type '%s'" % ltype)

        inputs = map(eval_, layer.get("input", []))
        params = dict( [ (k,eval_(v)) for k,v in layer.get("param", {}).items() ] )
        return constructor(name, inputs, params)
    
    def make_loss_(loss):
        name = loss.get("name")
        ltype = loss.get("type", None)
        if not ltype:
            raise_("loss must be explicitly assigned type")

        constructor = config["constructor_map"].get(ltype, None)
        if not constructor:
            raise_("loss type '%s' not supported" % ltype)
        
        inputs = map(eval_, loss.get("input", []))
        params = dict([ (k,eval_(v)) for k,v in loss.get("param", {}).items()])
        return constructor(name, inputs, params)
    
    def wrap_data_(data):
        if data is None:
            return None
        elif isinstance(data, numpy.ndarray):
            return NNData(data)
        else:
            raise_("type '%s' not supported for input data" % type(data))
    
    def make_update_(loss, ws, params):
        mappings = {
            "sgd" : SGDUpdate
        }
        gradtype = params.get("method","sgd")
        return mappings[gradtype](loss, ws, params)
      
    #输入：dim, dtype
    #输出：NNDataType
    def make_type_(dim, dtype):
        if dim>0:
            return NNArrayType(dim, dtype)
        elif dtype in ["float", "float32", "float64"]:
            return NNFloatType()
        elif dtype in ["int", "int32", "int64"]:
            return NNIntType()
        else:
            raise_("unsupported data type '%s'" % dtype)

    def check_name_(s):
        if not isinstance(s, str) or s=="":
            return False
        elif s in nn.core["values"]: 
            raise_("duplicate definition '%s'" % s)
        else: 
            return True
        
    nn = NNBase()
    config = global_config

    if not isinstance(data_dict, dict):
        raise_("data_dict should be a dictionary")

    #parameters
    for item in d.get("param", []):
        name = item.get("name","")
        if not check_name_(name): 
            continue
        value = item.get("value", None)
        if not value: 
            continue
        nn.core["values"][name] = eval_(value)

    #inputs
    for item in d.get("input", []):
        name = item.get("name","")
        if not check_name_(name): 
            continue
        dim = item.get("dim")
        dtype = item.get("type","float32")
        value = make_input_(dim, dtype)
        nn.core["values"][name] = value
        nn.core["inputs"][name] = value
        nn.core["data"][name] = wrap_data_( data_dict.get(name, None) )

    #weights
    for item in d.get("weight", []):
        name = item.get("name","")
        if not check_name_(name): 
            continue
        size = item.get("size",[])
        init = item.get("init", "random")
        dtype = item.get("type","float32")
        value = make_shared_(size, init, dtype)
        nn.core["values"][name] = value
        nn.core["weights"][name] = value
        if item.get("update", True):
            nn.core["learn"][name] = value

    #layers
    for item in d.get("layer", []):
        name = item.get("name","")
        if not check_name_(name): 
            continue
        nn.core["values"][name] = make_layer_(item)
    
    #losses 
    for item in d.get("loss", []):
        name = item.get("name","")
        if not check_name_(name): 
            continue
        value = make_loss_(item)
        nn.core["values"][name] = value
    
    #training
    item = d.get("training", None)
    if not item: 
        raise_("missing training section")
    if not item.has_key("loss"): 
        raise_("missing loss function")
    nn.core["loss"] = eval_( item["loss"] )
    nn.core["updates"] = make_update_(
        nn.core["loss"], 
        nn.core["learn"],
        {
            "learning_rate" : item.get("learning_rate",0.1),
            "method" : item.get("method", "sgd")
        }
    )    
    nn.core["test_info"] = [ (x, eval_(x)) 
                              for x in filter(lambda x:isinstance(x,str), item.get("test_info",[]))]
    nn.core["train_info"] = [ (x, eval_(x)) 
                              for x in filter(lambda x:isinstance(x,str), item.get("train_info",[]))]
    return nn

###################################################################################################################    

class NNData:
    def __init__(self, data):
        self.data = theano.shared(value=data, borrow=True)
    def get(self):
        return self.data.get_value()
    def get_wrap(self):
        return self.data

class NNValue:
    def node(self, ref=None):
        return self
    def value(self): 
        return self.v
    def is_durable(self):
        if not hasattr(self, "durable"):
            setattr(self, "durable", False)
        return self.durable

class NNScalar(NNValue):
    pass

class NNScalarInt(NNScalar):
    pass

class NNScalarInt64(NNScalarInt):
    def __init__(self, v=None):
        if v is None:
            self.v = T.scalar()
            self.durable = False
        else:
            self.v = v
            self.durable = True

class NNScalarFloat32(NNScalar):
    def __init__(self, v=None):
        if v is None:
            self.v = T.scalar()
            self.durable = False
        else:
            self.v = v
            self.durable = True

class NNArray(NNValue):
    def value(self):
        if isinstance(self.v, NNData):
            return self.v.get_wrap()
        else:
            return self.v
    def size(self):
        return self.dims

class NNArrayInt64(NNArray):
    def __init__(self, dims, v=None):
        self.dims = dims
        if v:
            self.v = v
            self.durable = True
        else:
            if len(dims)>4:
                raise_("NNArray support at most 4Darray temporarily")
            self.v = [T.vector, T.matrix, T.tensor3, T.tensor4][len(dims)-1]()
            self.durable = False

class NNArrayFloat32(NNArray):
    def __init__(self, dims, v=None):
        self.dims = dims
        if v:
            self.v = v
            self.durable = True
        else:
            if len(dims)>4:
                raise_("NNArray support at most 4Darray temporarily")
            self.v = [T.vector, T.matrix, T.tensor3, T.tensor4][len(dims)-1]()
            self.durable = False
###################################################################################################################

class Layer(NNValue):pass

class Loss(NNValue):pass

class LogisticLayer(Layer):
    def __init__(self, name, inputs, params):
        
        self.name = name

        if len(inputs)<2: 
            raise_("logistic layer '%s' need more inputs: [x, W, b]" % self.name)
        if not all( [isinstance(x, NNValue) for x in inputs] ): 
            raise_("inputs for layer '%s' should be of 'NNValue' type" % self.name)
        
        self.x = inputs[0]
        self.W = inputs[1]

        if len(self.x.size()) != 2:
            raise_("input data for logistic layer '%s' should be of dimension 2" % self.name)
        if len(self.W.size()) != 1: 
            raise_("weight vector for logistic layer '%s' should be of dimension 1" % self.name)

        if len(inputs)>2: 
            self.b = inputs[2]
        else:
            self.b = NNScalarFloat32(0.)

        self.y = NNArrayFloat32( v = T.nnet.sigmoid(T.dot(self.x.value(), self.W.value()) + self.b.value()),
                            dims = [self.x.size()[0]])
    
    def value(self):
        return self.y.value()

    def node(self, ref=None):
        if ref==None: 
            return self.y
        else: 
            raise_("logistic layer '%s' has no attribute '%s'" % (self.name, ref))

class SoftmaxLayer(Layer):
    def __init__(self, name, inputs, params):
        self.x = inputs[0]
        self.W = inputs[1]
        self.b = 0.
        if len(inputs)>2: self.b = inputs[2]
        self.y = T.nnet.softmax(T.dot(self.x, self.W) + self.b)
    def get_output(self):
        return self.y

#句子卷积层
#输入：[ data: batch_size * sentence_length * wordvec_length,
#        kernel: kernel_depth * 1 * kernel_width * wordvec_length,
#        mask: batch_size * sentence_length ]
#输出：batch_size * (sentence_length-kernel_width+1) * kernel_depth
class SentenceConvLayer(Layer):
    def __init__(self, name, inputs, params):
        self.name = name
        if len(inputs)!=3: 
            raise_("sentence convolution layer '%s' need 3 inputs: [data, kernel, mask]" % self.name)
        if not all( [isinstance(x, NNArray) for x in inputs] ): 
            raise_("inputs for layer '%s' should be of 'NNValue' type" % self.name)

        self.data, self.mask, self.kernel = inputs
        
        if len(self.data.size()) != 3: 
            raise_("input data for convolution layer '%s' should be of dimension 3" % self.name)
        if len(self.kernel.size()) != 4: 
            raise_("kernel for convolution layer '%s' should be of dimension 4" % self.name)
        if not all([isinstance(x,int) for x in self.kernel.size()]):
            raise_("kernel shape unknown for convolution layer '%s'" % self.name)
        if len(self.mask.size()) != 2: 
            raise_("data mask for convolution layer '%s' should be of dimension 2" % self.name)

        raw_conv = T.nnet.conv.conv2d(input = self.data.value().dimshuffle(0,'x',1,2), 
                                      filters = self.kernel.value(),
                                      filter_shape = self.kernel.size())
        
        kernel_width = self.kernel.size()[2]
        if kernel_width==1:
            conv_mask = self.mask.value()
        elif kernel_width>1:
            conv_mask = self.mask.value()[:,:1-kernel_width]
        else:
            raise_("kernel width of convolution layer '%s' should be at least 1" % self.name)

        reduced_conv = T.nnet.sigmoid(T.sum(raw_conv, axis=3)).dimshuffle(0,2,1) * conv_mask.dimshuffle(0,1,'x')

        data_size = self.data.size()
        trim_sentence_length = None if not data_size[1] else data_size[1]-kernel_width+1
        self.out = NNArrayFloat32( v = reduced_conv,
                    dims = [data_size[0], trim_sentence_length, data_size[2]] )

    def value(self):
        return self.out.value()

    def node(self, ref=None):
        if ref==None: 
            return self.out
        else: 
            raise_("convolution layer '%s' has no attribute '%s'" % (self.name, ref))

#MaxPooling层
#输入：[ data: batch_size * X * Y ]
#输出：batch_size * Y
class MaxPoolingLayer(Layer):
    def __init__(self, name, inputs, params):
        
        self.name = name
        if len(inputs) != 1: 
            raise_("max-pooling layer '%s' requires exactly 1 input" % self.name)
        if not all( [isinstance(x,NNArray) for x in inputs] ): 
            raise_("inputs for layer '%s' should be of 'NNArray' type" % self.name)
        
        self.data = inputs[0]
        if len(self.data.size()) != 3: 
            raise_("data for max-pooling layer '%s' should be of dimension 3" % self.name)
        
        pooling = T.max(self.data.value(), axis=1)
        self.out = NNArrayFloat32(v = pooling, 
                        dims = [self.data.size()[0], self.data.size()[2]])

    def node(self, ref=None):
        if ref==None: 
            return self.out
        else: 
            raise_("pooling layer '%s' has no attribute '%s'" % (self.name, ref))

    def value(self):
        return self.out.value()


class BinaryCrossEntropyLoss(Loss):
    def __init__(self, name, inputs, params):
        self.name = name

        if len(inputs) != 2: 
            raise_("BinaryCrossEntropy loss need inputs: [predict, label]")
        if not all( [isinstance(x, NNValue) for x in inputs] ): 
            raise_("inputs for loss '%s' should be of 'NNValue' type" % self.name)
        
        self.x, self.y = inputs[0], inputs[1]

        if len(self.x.size())!=1 or len(self.y.size())!=1:
            raise_("input datas for BinaryCrossEntropy loss should be of dimension 1")
        
        x, y = self.x.value(), self.y.value()
        loss = -T.mean( x*T.log(y) + (1-x)*T.log(1-y) )
        predict = T.switch(T.gt(x,0.5), 1, 0)
        self.loss = NNScalarFloat32(loss)
        self.predict = NNArrayInt64(dims=self.x.size()[:1], v=predict)

    def node(self, ref=None):
        if ref==None: 
            return self
        elif ref=="predict":
            return self.predict
        else: 
            raise_("BinaryCrossEntropy loss has no attribute '%s'" % ref)

    def value(self):
        return self.loss.value()

        return self.loss

class Update(NNValue):

    def node(self, ref=None):
        raise_("node() not supported for Update temporarily")

    def value(self):
        raise_("value() not supported for Update temporarily")

class SGDUpdate(Update):
    def __init__(self, loss, ws, params={}):
        if not isinstance(loss, NNValue):
            raise_("invalid loss type for SGDUpdata: %s" % loss)
        self.loss = loss
        self.grads = dict([
            (name, T.grad(self.loss.value(), ws[name].value()))
                for name in ws
        ])
        self.updates = map(
            lambda (w,g): (w, w-params.get("learning_rate",0.1)*g), 
            [ (ws[n].value(), self.grads[n]) for n in ws ]
        )

    def get(self):
        return self.updates

#################################################################################################


#######################################################################################################

class NNBase:

    def __init__(self):
     
        self.core = {

            #训练函数的实现
            "train_func" : None,

            #测试函数的实现
            "test_func" : None,

            #输入 :{str->NNValue}
            "inputs" : {},

            #参数 :{str->NNValue}
            "weights" : {},

            #待学习的参数 :{str->NNValue}
            "learn" : {},

            "updates" : None,

            #训练函数的输出 :{str->NNValue}
            "train_info" : None,

            #测试函数的输出 :{str->NNValue}
            "test_info" : None,
        
            #数据 :{str->NNValune}
            "data" : {},
        
            #中间数据 :{str->NNValue}
            "values" : {},

            "loss" : None
        }

    def __make_train_test(self):
        
        i,j = T.lscalar(), T.lscalar()

        #不必检查data是否存在
        givens = dict([
            ( v.value(), self.core["data"][name].get_wrap()[i:j] ) 
                for name,v in self.core["inputs"].items()
        ])

        self.core["train_func"] = theano.function(
            inputs = [i,j],
            givens = givens,
            updates = self.core["updates"].get(),
            outputs = [v.value() for (n,v) in self.core["train_info"]]
        )

        self.core["test_func"] = theano.function(
            inputs = [i,j],
            givens = givens,
            outputs = [v.value() for (n,v) in self.core["test_info"]]
        )

    def __raise(self, text, level=0):
        raise Exception(text)
        
    def __check(self):
        #检查数据
        for name in self.core["inputs"]:
            if not self.core["values"].has_key(name):
                self.__raise("[NNBase] input '%s' not found" % name)
            if not self.core["data"].has_key(name):
                self.__raise("[NNBase] missing input data '%s', use set_data() to set it" % name)
        #构造训练及测试函数
        if self.core["train_func"]==None:
            self.__make_train_test()

    def train(self, beg, end):
        self.__check()
        return self.core["train_func"](beg, end)

    def test(self, beg, end):
        self.__check()
        return self.core["test_func"](beg, end)

##########################################错误处理##############################################
def raise_(msg, level=0):
    raise Exception("[Error] " + msg)
