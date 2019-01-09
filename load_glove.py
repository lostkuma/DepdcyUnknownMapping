FILENAME = "glove.twitter.27B/glove.twitter.27B.100d.txt"
import numpy as np


class Embedding(object):
	def __init__(self, text=None, vec=None):
		self.text = text
		self.vec = vec
		
	def __repr__(self):
		return "{}, {}".format(self.text, self.vec)
	
	def get_text(self):
		return self.text
	
	def get_vector(self):
		return self.vec
	
	def vec_dim(self):
		return len(self.vec)


class Glove(object):
	def __init__(self, pre_trained=list()):
		self.pre_trained = pre_trained
		self.text_to_embedding = dict()
		for embedding in self.pre_trained:
			self.text_to_embedding[embedding.get_text()] = embedding
			
	def __repr__(self):
		return "<Glove_object num_tokens.{} vec_dim.{}>".format(len(self.pre_trained), self.pre_trained[0].vec_dim())
	
	def get_vector(self, text):
		if text not in self.text_to_embedding:
			print("'{}' not in the model".format(text))
			return None
		return self.text_to_embedding[text].get_vector()
	
	def similarity(self, token1, token2):
		vec1 = self.get_vector(token1)
		vec2 = self.get_vector(token2)
		if len(vec1) != len(vec2):
			print("token1: {}, token2: {} not having same dimention".format(token1, token2))
			return 
		norm1 = np.array(vec1) / np.linalg.norm(vec1)
		norm2 = np.array(vec2) / np.linalg.norm(vec2)
		return np.dot(norm1, norm2)

	def similarity_embedding(self, embedding1, embedding2):
		try: 
			embedding1 = embedding1.get_vector()
		except AttributeError: 
			pass
		try: 
			embedding2 = embedding2.get_vector()
		except AttributeError: 
			pass
		if len(embedding1) != len(embedding2):
			print("token1: {}, token2: {} not having same dimention".format(token1, token2))
			return
		norm1 = np.array(embedding1) / np.linalg.norm(embedding1)
		norm2 = np.array(embedding2) / np.linalg.norm(embedding2)
		return np.dot(norm1, norm2)

	def most_similar(self, text, topn=10):
		index_to_dot_value = dict()
#        count = 0
		
		keys = list(self.text_to_embedding.keys())
		for i, key in enumerate(keys):
#            count += 1
#            if count % 10000 == 0:
#                print("{} checked".format(count))
			index_to_dot_value[i] = self.similarity(text, self.text_to_embedding[key].get_text())
			
		topn_index = sorted(index_to_dot_value.items(), key=lambda x: x[1], reverse=True)[1:topn+1]
		topn_values = list()
		for pair in topn_index:
			topn_values.append((keys[pair[0]], pair[1]))
		return topn_values

	def most_similar_token(self, target, topn=10):
		key_to_dot_value = dict()
		for key in self.text_to_embedding.keys():
			embedding_in_model = self.text_to_embedding[key].get_vector()
			key_to_dot_value[key] = self.similarity_embedding(target, embedding_in_model)

		topn_index = sorted(key_to_dot_value.items(), key=lambda x: x[1], reverse=True)[1:topn+1]
		topn_words = list()
		for pair in topn_index:
			topn_words.append((pair[0], pair[1]))
		return topn_words


def LoadPretrainedGlove(file=FILENAME):
	pre_trained = list()
	with open(file, "r", encoding="utf-8") as textfile:
		row_count = 0
		line = textfile.readline()
		while line:
			row_count += 1
			line = line.split(' ')
			text = line.pop(0)
			try:
				vector = list(map(float, line))
			except ValueError:
				print("Error loading '{}' in row {} from file '{}'.".format(text, row_count, FILENAME))
				continue
			pre_trained.append(Embedding(text, vector))
			line = textfile.readline()

	model = Glove(pre_trained)
	return model
