
Similarity, Comparison Building Block (DL) 
-------------------------------------------
Retail industry faces a difficult situation as retailers would like to understand the products that are very similar to each other in order to evaluate which product is better off not promoting from the similar products in the same week to increase sales activity and profit margins.

Finding similar images, sentences are a very common use case in many industries.  Layer weight sharing(reuse) can help in scenarios where comparison, gauging similarities is required. Re-using a layer with same weights can help in these cases.
Most architectures where comparison or similarity is the scenario, will make use of layer weight sharing which becomes the basic architecture building block. 
Finding similar sentences
Walking through an example accessing the semantic similarity between 2 sentences. The model will ideally have 2 inputs (2 sentences to compare) and outputs a score of 0 and 1. 0 means related and vice versa.  
In this setup two input sentences are interchangeable, as semantic similarity is symmetrical relationship i.e. the similarity of A and B is identical to similarity of B and A. For this reason, there is no need to make two independent models for processing each input sentences. Rather it will appropriate to process both with a single LSTM layer (its weights) are learned based on both inputs simultaneously. This is called Siamese twins or a shared LSTM.
Taken a simple example to code further find the complete code here. Below is the network diagram of the same.

The sentences are put through an embedding layer(word2vec). The same bidirectional LSTM layers are used across left and right sentence merged by a concatenation operation and finally a Dense layer with sigmoid activation function for classifying a match 1 or non-match 0. This ideally becomes the building block for most comparators. Most architecture will use the similar construct for more complex scenarios.


Note: Instantiating and training the model: when you train such a model, the weights of the bidirectional LSTM layer are updated based on both inputs.
The data used is https://www.kaggle.com/c/quora-question-pairs/data. 
The validation accuracy is in the range of 75% the same can be improved.
Finding similar Images
Conceptually this is like comparing text except since there are images involved, CNN is choice of model as handles images better. The architecture consists of a similar building block consisting of a series of CNN with pooling layers one can use Inception V3 for better accuracy.
The network diagram for the same is indicated below
 
The example code consists of identifying similar images https://zenodo.org/record/1049137#.Xc-G-VczZPY. Data augmentation techniques such image generation is used to create similar images.
The validation accuracy is about 80%.
The code for the same can be found here.


