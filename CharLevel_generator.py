import torch
import torch.nn as nn
from math import log
import os


data = open('input.txt', 'r').read() # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print ('data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_index = { ch:i for i,ch in enumerate(chars) }
index_to_char = { i:ch for i,ch in enumerate(chars) }

BATCH_SIZE = 5
HIDDEN_SIZE = 128 # size of hidden layer of neurons
LAYER_NUMBER = 3
optimizer_lr = 1.0e-3
SEQ_LENGTH = 25 # number of steps to unroll the RNN for


# If we have a GPU available, we'll set our device to GPU
if torch.cuda.is_available():
	device = torch.device("cuda")
	print("GPU is available")
else:
	device = torch.device("cpu")
	print("GPU not available, CPU used")


# Last forward layer is linear. CE_Loss and argmax used for optimization
class RNN(nn.Module):
	def __init__(self, input_size, HIDDEN_SIZE, 
				 LAYER_NUMBER, output_size, BATCH_SIZE, act='tanh', 
				 dropout_value=0.015):
		super(RNN, self).__init__()
		self.BATCH_SIZE = BATCH_SIZE
		self.HIDDEN_SIZE = HIDDEN_SIZE
		self.LAYER_NUMBER = LAYER_NUMBER

		self.i2h1 = nn.Linear(input_size + HIDDEN_SIZE, HIDDEN_SIZE)
		self.norm1 = nn.LayerNorm(normalized_shape=HIDDEN_SIZE)
		self.act1 = nn.Tanh() if act=='tanh' else nn.ReLU()
		self.drop1 = nn.Dropout(dropout_value)

		self.h12h2 = nn.Linear(HIDDEN_SIZE + HIDDEN_SIZE, HIDDEN_SIZE)
		self.norm2 = nn.LayerNorm(normalized_shape=HIDDEN_SIZE)
		self.act2 = nn.Tanh() if act=='tanh' else nn.ReLU()
		self.drop2 = nn.Dropout(dropout_value)

		self.h22h3 = nn.Linear(HIDDEN_SIZE + HIDDEN_SIZE, HIDDEN_SIZE)
		self.norm3 = nn.LayerNorm(normalized_shape=HIDDEN_SIZE)
		self.act3 = nn.Tanh() if act=='tanh' else nn.ReLU()
		self.drop3 = nn.Dropout(dropout_value)

		self.h2o = nn.Linear(HIDDEN_SIZE, output_size)
		
	def forward(self, input, hidden):
		hidden1 = hidden[0]
		hidden2 = hidden[1]
		hidden3 = hidden[2]
		input_combined = torch.cat((input, hidden1), 1)

		hidden1 = self.i2h1(input_combined)
		hidden1 = self.norm1(hidden1)
		hidden1 = self.act1(hidden1)
		hidden1 = self.drop1(hidden1)

		hidden1_combined = torch.cat((hidden2, hidden1), 1)

		hidden2 = self.h12h2(hidden1_combined)
		hidden2 = self.norm2(hidden2)
		hidden2 = self.act2(hidden2)
		hidden2 = self.drop2(hidden2)

		hidden2_combined = torch.cat((hidden3, hidden2), 1)

		hidden3 = self.h22h3(hidden2_combined)
		hidden3 = self.norm3(hidden2)
		hidden3 = self.act3(hidden3)
		hidden3 = self.drop3(hidden3)

		output = self.h2o(hidden3)

		return output, [hidden1, hidden2, hidden3]

	def init_hidden(self, is_batch=True):
		return [torch.zeros(self.BATCH_SIZE if is_batch else 1,
							self.HIDDEN_SIZE)
					for _ in range(self.LAYER_NUMBER)]


# input is a list with indexes of symbols in vocabulary
def inputTensor(line):
	tensor = torch.zeros(1, len(line), vocab_size)
	for i in range(len(line)):
		index_of_current_symbol = line[i]
		tensor[0][i][index_of_current_symbol] = 1
	return tensor


criterion = nn.CrossEntropyLoss() # LogSoftmax + NLLLoss
rnn=RNN(vocab_size, HIDDEN_SIZE, LAYER_NUMBER, vocab_size, BATCH_SIZE)
optimizer = torch.optim.Adam(rnn.parameters(), lr=optimizer_lr)
rnn = rnn.to(device)


# Don't forget to encode inputs and targets!
def train(rnn_model, hprev, input_line_tensor, target_line_tensor):
	# target_line_tensor.unsqueeze_(-1)
	hidden = list(map(lambda h: h.to(device), hprev))
	rnn_model.zero_grad()
	total_loss = 0

	# iterate through sequence dim
	for seq_token in range(input_line_tensor.size(1)):
		output, hidden = \
			rnn_model(input_line_tensor[:,seq_token].to(device), hidden)
		hidden = list(map(lambda h: h.to(device), hidden))
		output = output.to(device)

		loss = criterion(output, target_line_tensor[:,seq_token])
		total_loss += loss

	total_loss.backward()
	nn.utils.clip_grad_value_(rnn_model.parameters(), 5)
	optimizer.step()

	return [h.data for h in hidden], total_loss.item() / input_line_tensor.size(1)


def sample(rnn_model, hidden, letter_index, sample_length, decoder_vocab):
	# type(letter_index) == list, contains 1 sybmol index
	with torch.no_grad():  # no need to track history during sampling
		input_ = inputTensor(letter_index).to(device)
		# hidden = rnn_model.init_hidden(is_batch=False)
		hidden = list(map(lambda h: h.to(device), hidden))

		output_indices = []
		output_indices.append(letter_index[0])

		for _ in range(sample_length):
			output, hidden = rnn_model(input_[0].to(device), hidden)
			hidden = list(map(lambda h: h.to(device), hidden))

			# select symbol with highest probability in output
			topv, topi = output.topk(1)
			topi = topi[0][0]
			topi = int(topi)
			output_indices.append(topi)
			input_ = inputTensor([topi])

		return ''.join(decoder_vocab[index] for index in output_indices)


# Load checkpoint
if os.path.exists('setting/parameters.pt'):
	with torch.load('setting/parameters.pt') as checkpoint:
		rnn.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		char_to_index, index_to_char = checkpoint['vocabulary']
		n, p, smooth_loss = checkpoint['support']
		hidden_values = checkpoint['hidden_values']

	rnn.train()
else:
	if not os.path.exists('setting'): os.mkdir('setting')
	n, p = 0, 0
	smooth_loss = -log(1.0/vocab_size)*SEQ_LENGTH # loss at iteration 0
if not os.path.exists('samples'): os.mkdir('samples')

while True:
	if (p+SEQ_LENGTH*BATCH_SIZE+1 >= len(data)) or (n == 0):
		hidden_values = rnn.init_hidden(is_batch=True)
		hidden_values = list(map(lambda h: h.to(device), hidden_values))
		p = 0

	inputs, targets = list(), list()
	for _ in range(BATCH_SIZE):
		inputs.append( inputTensor(
						[char_to_index[ch] for ch in data[p:p+SEQ_LENGTH]])
		)
		targets.append( torch.LongTensor(
						[char_to_index[ch] for ch in data[p+1:p+SEQ_LENGTH+1]]).reshape((1,-1))
		)
		p+=SEQ_LENGTH*BATCH_SIZE
	inputs = torch.cat(inputs).to(device)
	targets = torch.cat(targets).to(device)

	hidden_values, loss_v = \
		train(rnn, hidden_values, inputs, targets)
	hidden_values = list(map(lambda h: h.to(device), hidden_values))
	smooth_loss = smooth_loss * 0.999 + loss_v * 0.001

	if n % 100 == 0:
		# sample uses inputTensor function, 
		# so we should wrap input index in a list
		txt = sample(rnn,
					 list(map(lambda h: h[-1].reshape((1,-1)), hidden_values)),
					 letter_index=[char_to_index[data[p]]],
					 sample_length=200,
					 decoder_vocab=index_to_char)
		print ('----\n %s \n----' % (txt, ))

		# print('iter %d(k), loss: %f' % (int(n/1000), smooth_loss))
		print('iter %d, loss: %f' % (n, smooth_loss))

	# Make checkpoint
	if n % 25000 == 0:
		torch.save({
				'model_state_dict': rnn.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),
				'vocabulary': (char_to_index, index_to_char),
				'support': (n, p, smooth_loss),
				'hidden_values': hidden_values
				}, 'setting/parameters.pt')
		print('checkpoint\'s done!')

	if n % 50000 == 0:
		with open('samples/sample_'+str(int(n/1000))+'(k).txt','w') as example:
			txt = sample(rnn,
						 list(map(lambda h: h[-1].reshape((1,-1)), hidden_values)),
						 [char_to_index[data[p]]], 1000, index_to_char)
			example.write(txt)

	n+=1
