import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class SkipGram(nn.Module):
	# the skip gram model
	def __init__(self, vocabulary_size, embedding_dim):
		super(SkipGram, self).__init__()
		self.embedding_dim = embedding_dim
		self.vocabulary_size = vocabulary_size
		# initialize the embedding layer randomly with embedding dimension and vocab size
		# one column of the matrix is an embedding for a word
		self.embeddings = nn.Embedding(self.vocabulary_size, self.embedding_dim)

	def forward(self, x, y_true):
		center = self.embeddings(x).view(1, -1)
		context = self.embeddings(y_true).view(1, -1)
		out = torch.matmul(cneter, torch.t(context))
		out = F.logsigmoid(out)
		return out

	def get_word_vector(self, word_idx):
		word = Variable(torch.LongTensor([word_idx]))
		return self.embeddings(word).view(1, -1)

	def save_model(self, file_name, index2word):
		# write trained embeddings to file
		with open(file_name, 'w', encoding='utf-8') as textfile:
			for index in range(len(self.embeddings.weight.data)):
				token = index2word(index)
				embedding = ' '.join(self.embeddings.weight.data[index])
				textfile.write('{} {}\n'.format(token, embedding))


class CBOW(nn.Module):
	# the continuous bag of words model
	def __init__(self, vocabulary_size, embedding_dim, context_size):
		super(CBOW, self).__init__()
		self.embedding_dim = embedding_dim
		self.vocabulary_size = vocabulary_size
		self.context_size = context_size
		# initialize the embedding layer randomly with embedding dimension and vocab size
		# one column of the matrix is an embedding for a word
		self.embeddings = nn.Embedding(self.vocabulary_size, self.embedding_dim)
		# input size is context size * 2 (sides of center) * embedding size
		self.linear1 = nn.Linear(self.context_size * 2 * self.embedding_dim, 256)
		self.linear2 = nn.Linear(256, self.vocabulary_size)

	def forward(self, x):
		out = self.embeddings(x).view(1, -1)
		out = self.linear1(out)
		out = F.relu(out)
		out = self.linear2(out)
		out = F.log_softmax(out, dim=1)
		return out 

	def get_word_vector(self, word_idx):
		word = Variable(torch.LongTensor([word_idx]))
		return self.embeddings(word).view(1, -1)

	def save_model(self, file_name, index2word):
		# write trained embeddings to file
		with open(file_name, 'w', encoding='utf-8') as textfile:
			for index in range(len(self.embeddings.weight.data)):
				token = index2word(index)
				embedding = ' '.join(self.embeddings.weight.data[index])
				textfile.write('{} {}\n'.format(token, embedding))


class NN(nn.Module):
	# the input dim should be padded based on the maximum dependency one token has
	# the output dim is the same with the dim of the word embeddings used 
	# the model predict the word embedding of a word given all the dependency info the word has
	def __init__(self, input_dim, output_dim):
		super(NN, self).__init__()
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.linear1 = nn.Linear(self.input_dim, 512)
		self.linear2 = nn.Linear(512, self.output_dim)

	def forward(self, x):
		out = self.linear1(x)
		out = F.relu(out)
		#out = torch.tanh(out)
		out = self.linear2(out)
		#out = F.log_softmax(out, dim=0)
		return out






