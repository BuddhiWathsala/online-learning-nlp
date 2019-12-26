# Online Learning for Solving Data Availability Problem in Natural Language Processing
## Problem
Machine learning (ML) models need large data sources to obtain better accuracies. However, obtaining a large data source at once take many resources. In practical we obtain large data sources incrementally, often as a several mini-batches. The batch learning techniques in ML need to retrain the model from scratch when we attain a mini-batch. This problem exists in most of the ML models which follow batch learning techniques. This problem also exists in Natural Language Processing (NLP) tasks. In this project we propose solutions to this problem using online machine learning techniques. Online machine learning techniques has tha capability of training incrementally using the data obtain as a stream. 

You can directly apply these ML models into Named Entity Recognition (NER) and Part of Speech (POS) tagging tasks in any language. One of the major advatage of this solution is we can apply it to any NER and POS tagging task in any language.
Moreover, we can apply this solution to any structured prediction task with slight modifications to the code. These things are discussed in further.

## Proposed Solutions
This project proposed two main solutions.
  1. Online Conditional Random Fiels (CRF) Model
  2. Bidirectional Long Short Term Memory-Conditional Random Fiels (LSTM-CRF) Model

## Installation 
1. First you need Python 3.X version installed in your computer.
1. Then you need to install `tensorflow`, `keras,` and `keras_contrib` libraries. You can install these librarie easily using `pip` or other methods.
    1. `tensorflow` installation. (Ref: https://www.tensorflow.org/install/)
        * `pip install tensorflow`
    1. `keras` installation. Better to use `keras 2.2.2` version. (Ref: https://keras.io/)
        * `pip install -q keras==2.2.2`
    1. `keras_contrib` installation. (Ref: https://github.com/keras-team/keras-contrib)
        * `pip install git+https://www.github.com/keras-team/keras-contrib.git`
        
## Dataset
Mainly, we apply these models into NER and POS tagging. For that the dataset should adhere to the CoNLL-2003 data format. The dataset sholud be a text file which have main two columns. First column contained the word and the second column contained the corresponding tag of that word (NE tag or POS tag). The text file contained empty lines to indcate sentence boundries.

You can apply this online learning models to other structured prediction task. In order to do that you can change the dataset according to that. Instead of using words in the dataset you can directly replace the word with the feature vector corresponding to your task. And the NE or POS tag can be replaced with the tag which relavent to the feature vector.

After that you can store the datafile in the `resources/` directory. Then add the file name to the variable `DATA_FILE` in code file that you are going to use (`crf.py` or `deep-crf.py`).

## Running
Move to the `src/` directory and run the code using terminal with following arguments.

`python <arg1> <arg2> <arg3> <arg4> <arg5> <arg6>`
  1. `arg1`: True if you train the model. False if you test the model.
  1. `arg2`: Name of the data file.
  1. `arg3`: Number of epochs to be execute.
  1. `arg4`: Splitting factor of training and testing (ex: 0.1 for 90:10)
  1. `arg5`: Splitting factor of training and validation (ex: 0.1 for 90:10)
  1. `arg6`: True if you initially train the model. False indicates you train the model in an intermediate incremental step.

First two arguments are compulsory. Others are optional. When you miss the optional paramenters, those paramentes will take default values.

## Publication

Find the research publication [here](http://ceur-ws.org/Vol-2521/paper-04.pdf)
