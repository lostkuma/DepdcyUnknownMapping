import json
import subprocess
import time
from pycorenlp import StanfordCoreNLP

CORRECTED_DATA_PATH = '/Users/kuma/Desktop/Courses/BME 595/Project/fce-released-dataset/dataset/correct.sentence.txt'
ORIGINAL_DATA_PATH = '/Users/kuma/Desktop/Courses/BME 595/Project/fce-released-dataset/dataset/original.sentence.txt'
SPELLING_ERROR_DATA_PATH = '/Users/kuma/Desktop/Courses/BME 595/Project/fce-released-dataset/dataset/original.spelling.tags.txt'
CORRECTED = False 

def LoadSentences(filename):
	count_sentences = 0
	sentences = []
	# the file contains sentences on each line
	with open(filename, 'r', encoding='utf-8') as textfile:
		line = textfile.readline()
		while line:
			count_sentences += 1
			line = line.strip('\n')
			sentences.append(line)
			line = textfile.readline()
	return sentences, count_sentences

def main():
	# parse each sentence and obtain the basic and enhanced dependencies
	# output json file with the dependency trees for each sentence

	dep_output = []
	tokenize_output = []

	if CORRECTED:
		sentences, tnum_sentences = LoadSentences(CORRECTED_DATA_PATH)
	else:
		sentences, tnum_sentences = LoadSentences(SPELLING_ERROR_DATA_PATH)

	server_subprocess = subprocess.Popen("java -mx4g -cp '*' edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000",
						shell=True, 
						stdout=subprocess.DEVNULL, 
						stderr=subprocess.DEVNULL)

	print('loading stanford core nlp dependency parser...')
	time.sleep(10)
	print('parser loaded. start parsing...')

	nlp = StanfordCoreNLP('http://localhost:9000')

	counter = 0
	skip_counter = 0

	for sent in sentences:
		counter += 1

		try:
			# tokenize and parse the sentence with dependency parser
			parsed = nlp.annotate(sent, properties={'annotators': 'tokenize,ssplit,depparse', 'outputFormat': 'json'})
			# print(parsed['sentences'][0]['enhancedPlusPlusDependencies'])
			dep_output.append(parsed['sentences'][0]['enhancedPlusPlusDependencies'])
			# list of sentences sliced with space
			tokenized = parsed['sentences'][0]['tokens']
			tokenized_sent = []
			for element in tokenized:
				tokenized_sent.append(element['word'])
			tokenize_output.append(tokenized_sent)

		except IndexError:
			# deal with rows that has nothing on it
			skip_counter += 1
			print('IndexError: sentence {} skipped'.format(counter))
			continue

		if counter % 500 == 0:
			print('  finished parsing {} / {} sentences'.format(counter, tnum_sentences))

	print('{} / {} sentences skipped'.format(skip_counter, tnum_sentences))

	if CORRECTED:
		with open('dependencies.corrected.txt', 'w', encoding='utf-8') as textfile:
			json.dump(dep_output, textfile) 
		with open('tokenized.corrected.txt', 'w', encoding='utf-8') as textfile:
			for sent in tokenize_output:
				for word in sent:
					textfile.write('{} '.format(word))
				textfile.write('\n')
	else:
		with open('dependencies.spelling.txt', 'w', encoding='utf-8') as textfile:
			json.dump(dep_output, textfile)
		with open('tokenized.spelling.txt', 'w', encoding='utf-8') as textfile:
			for sent in tokenize_output:
				for word in sent:
					textfile.write('{} '.format(word))
				textfile.write('\n')

	subprocess.Popen.kill(server_subprocess)


if __name__ == '__main__':
	main()
