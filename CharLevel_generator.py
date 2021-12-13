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

optimizer_lr = 1.0e-3
HIDDEN_SIZE = 128 # size of hidden layer of neurons
seq_length = 25 # number of steps to unroll the RNN for
batch_size = 100

# If we have a GPU available, we'll set our device to GPU
if torch.cuda.is_available():
	device = torch.device("cuda")
	print("GPU is available")
else:
	device = torch.device("cpu")
	print("GPU not available, CPU used")


# Last forward layer is linear. CE_Loss and argmax used for optimization
class RNN(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, act='tanh', dropout_value=0.015):
		super(RNN, self).__init__()
		self.hidden_size = hidden_size

		self.i2h1 = nn.Linear(input_size + hidden_size, hidden_size)
		self.norm1 = nn.LayerNorm(normalized_shape=hidden_size)
		self.act1 = nn.Tanh() if act=='tanh' else nn.ReLU()
		self.drop1 = nn.Dropout(dropout_value)

		self.h12h2 = nn.Linear(hidden_size + hidden_size, hidden_size)
		self.norm2 = nn.LayerNorm(normalized_shape=hidden_size)
		self.act2 = nn.Tanh() if act=='tanh' else nn.ReLU()
		self.drop2 = nn.Dropout(dropout_value)

		self.h22h3 = nn.Linear(hidden_size + hidden_size, hidden_size)
		self.norm3 = nn.LayerNorm(normalized_shape=hidden_size)
		self.act3 = nn.Tanh() if act=='tanh' else nn.ReLU()
		self.drop3 = nn.Dropout(dropout_value)

		self.h2o = nn.Linear(hidden_size, output_size)
		
	def forward(self, input, hidden1, hidden2, hidden3):
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

		return output, hidden1, hidden2, hidden3

	def initHidden(self):
		return torch.zeros(1, self.hidden_size)


# input is a list with indexes of symbols in vocabulary
def inputTensor(line):
	tensor = torch.zeros(1, len(line), vocab_size)
	for i in range(len(line)):
		index_of_current_symbol = line[i]
		tensor[0][i][index_of_current_symbol] = 1
	return tensor


criterion = nn.CrossEntropyLoss() # LogSoftmax + NLLLoss
rnn=RNN(vocab_size, HIDDEN_SIZE, vocab_size)
optimizer = torch.optim.Adam(rnn.parameters(), lr=optimizer_lr)
rnn = rnn.to(device)


# Don't forget to encode inputs and targets!
def train(rnn_model, hprev1, hprev2, hprev3, input_line_tensor, target_line_tensor):
	# target_line_tensor.unsqueeze_(-1)
	hidden1, hidden2, hidden3 = \
		hprev1.to(device), hprev2.to(device), hprev3.to(device)
	rnn_model.zero_grad()
	total_loss = 0

	# iterate through sequence dim
	for seq_token in range(input_line_tensor.size(1)):
		output, hidden1, hidden2, hidden3 = \
			rnn_model(input_line_tensor[:,seq_token].to(device), hidden1, hidden2, hidden3)
		hidden1, hidden2, hidden3 = hidden1.to(device), hidden2.to(device), hidden3.to(device)
		output = output.to(device)

		loss = criterion(output, target_line_tensor[:,seq_token])
		total_loss += loss

	total_loss.backward()
	nn.utils.clip_grad_value_(rnn_model.parameters(), 0.5)
	optimizer.step()

	return hidden1.data, hidden2.data, hidden3.data, total_loss.item() / input_line_tensor.size(1)


def sample(rnn_model, h1, h2, h3, letter_index, sample_length, decoder_vocab):
	# type(letter_index) == list, contains 1 sybmol index
	with torch.no_grad():  # no need to track history during sampling
		input_ = inputTensor(letter_index).to(device)
		# hidden = rnn_model.initHidden()
		hidden1, hidden2, hidden3 = h1.to(device), h2.to(device), h3.to(device)

		output_indices = []
		output_indices.append(letter_index[0])

		for _ in range(sample_length):
			output, hidden1, hidden2, hidden3 = \
				rnn_model(input_[0].to(device), hidden1, hidden2, hidden3)
			hidden1, hidden2, hidden3 = \
				hidden1.to(device), hidden2.to(device), hidden3.to(device)

			# select symbol with highest probability in output
			topv, topi = output.topk(1)
			topi = topi[0][0]
			topi = int(topi)
			output_indices.append(topi)
			input_ = inputTensor([topi])

		return ''.join(decoder_vocab[index] for index in output_indices)


# Load checkpoint
if os.path.exists('setting/parameters.pt'):
	checkpoint = torch.load('setting/parameters.pt')
	rnn.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	char_to_index, index_to_char = checkpoint['vocabulary']
	n, p, smooth_loss = checkpoint['support']
	hidden_values1 = checkpoint['hidden_values1']
	hidden_values2 = checkpoint['hidden_values2']
	hidden_values3 = checkpoint['hidden_values3']

	rnn.train()
else:
	if not os.path.exists('setting'): os.mkdir('setting')
	n, p = 0, 0
	smooth_loss = -log(1.0/vocab_size)*seq_length # loss at iteration 0
if not os.path.exists('samples'): os.mkdir('samples')

while True:
	if (p+seq_length*batch_size+1 >= len(data)) or (n == 0):
		hidden_values1 = [rnn.initHidden() for _ in range(batch_size)]
		hidden_values2 = [rnn.initHidden() for _ in range(batch_size)]
		hidden_values3 = [rnn.initHidden() for _ in range(batch_size)]

		hidden_values1 = torch.cat(hidden_values1).to(device)
		hidden_values2 = torch.cat(hidden_values2).to(device)
		hidden_values3 = torch.cat(hidden_values3).to(device)
		p = 0

	inputs, targets = list(), list()
	for _ in range(batch_size):
		inputs.append( inputTensor(
						[char_to_index[ch] for ch in data[p:p+seq_length]])
		)
		targets.append( torch.LongTensor(
						[char_to_index[ch] for ch in data[p+1:p+seq_length+1]]).reshape((1,-1))
		)
		p+=seq_length
	inputs = torch.cat(inputs).to(device)
	targets = torch.cat(targets).to(device)

	hidden_values1, hidden_values2, hidden_values3, loss_v = \
		train(rnn, hidden_values1, hidden_values2, hidden_values3, inputs, targets)
	hidden_values1, hidden_values2, hidden_values3 = \
		hidden_values1.to(device), hidden_values2.to(device), hidden_values3.to(device)
	smooth_loss = smooth_loss * 0.999 + loss_v * 0.001

	if n % 100 == 0:
		# sample uses inputTensor function, 
		# so we should wrap input index in a list
		txt = sample(rnn,
					 hidden_values1[-1].reshape((1,-1)), 
					 hidden_values2[-1].reshape((1,-1)), 
					 hidden_values3[-1].reshape((1,-1)), 
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
				'hidden_values1': hidden_values1,
				'hidden_values2': hidden_values2,
				'hidden_values3': hidden_values3
				}, 'setting/parameters.pt')
		print('checkpoint\'s done!')

	if n % 50000 == 0:
		with open('samples/sample_'+str(int(n/1000))+'(k).txt','w') as example:
			txt = sample(rnn,
						 hidden_values1[-1].reshape((1,-1)), 
						 hidden_values2[-1].reshape((1,-1)), 
						 hidden_values3[-1].reshape((1,-1)), 
						 [char_to_index[data[p]]], 1000, index_to_char)
			example.write(txt)

	n+=1
