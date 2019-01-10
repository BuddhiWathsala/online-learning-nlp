# online-learning-nlp
## Problem
Machine learning (ML) models need large data sources to obtain better accuracies. However, obtaining a large data source at once take many resources. In practical we obtain large data sources incrementally, often as a several mini-batches. The batch learning techniques in ML need to retrain the model from scratch when we attain a mini-batch. This problem exists in most of the ML models which follow batch learning techniques. This problem also exists in Natural Language Processing (NLP) tasks. In this project we propose solutions to this problem using online machine learning techniques. Online machine learning techniques has tha capability of training incrementally using the data obtain as a stream. 

You can directly apply these ML models into Named Entity Recognition (NER) and Part of Speech (POS) tagging tasks in any language. One of the major advatage of this solution is we can apply it to any NER and POS tagging task in any language.
Moreover, we can apply this solution to any structured prediction task with slight modifications to the code. These things are discussed in further.

## Proposed Solutions
This project proposed two main solutions.
  1. Online Conditional Random Fiels (CRF) Model
  2. Bidirectional Long Short Term Memory-Conditional Random Fiels(LSTM-CRF) Model

## Installation 
1. First you need Python 3.X version installed in your computer.
1. Then you need to install `tensorflow`, `keras,` and `keras_contrib` libraries. You can install these librarie easily using `pip` or other methods.
    1. `tensorflow` installation. (Ref: https://www.tensorflow.org/install/)
        * `pip install tensorflow`
    1. `keras` installation. Better to use `keras 2.2.2` version. (Ref: https://keras.io/)
        * `pip install -q keras==2.2.2`
    1. `keras_contrib` installation. (Ref: https://github.com/keras-team/keras-contrib)
        * `pip install git+https://www.github.com/keras-team/keras-contrib.git`
