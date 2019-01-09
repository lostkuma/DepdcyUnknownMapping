# Dependency Based Unkonwn Words Mapping
2018 Fall BME 595 Deep Learning Project
An exploration on mapping unkonwn tokens using known contexts based on word embedding. 


Glove pre-train embedding download (Twitter 27B): https://nlp.stanford.edu/projects/glove/ 

Stanford coreNLP download (3.9.2): https://stanfordnlp.github.io/CoreNLP/history.html

A Google drive folder containing all code, files, and source data: https://drive.google.com/open?id=1aC5iRMfFYfrxMC2MxqPVFVIyXCuGgc9T

	<<Google drive description>>

	• Dataset – the dataset folder, Treebank of learner English (Berzak et al., 2016)

		o raw.data.process.ipynb – jupyter notebook file for processing raw data

		o there are other source code files but they were not used in this project

		o original.spelling.tags.txt, correct.sentence.txt – the processed text files used in this project

	• train.ipynb – jupyter notebeook file for training the three models

	• test.ipynb – jupyter notebook file for testing the best model

	• models.py – containing a skip-gram model, CBOW model, and an NN model

	• load_glove.py – loading the raw Glove 27B 100d pre-trained model

	• dep_parsing.py – using Stanford coreNLP for dependency parsing and tokenization

	• tokenized.spelling.txt – tokenized test set

	• tokenized.corrected.txt – tokenized training set

	• dependencies.spelling.txt – all dependencies in the test set for the sentences

	• dependencies.correced.txt – all dependencies in the training set for the sentences

	• there are other files which I planned on using but ended up not using them in the folder

	• nn.softmax.best.model – model1 trained in this project

	• nn.tanh.best.model – model2 trained in this project

	• nn.nosoftmax.best.model – model3 trained in this project

	• Slides – the presentation slides for the project
