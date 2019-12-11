
'''

Rwiddhi Chakraborty
11/12/2019

tensorflow : 1.15.0
python: 3.6

'''

import numpy as numpy
import tensorflow as tf

from collections import Counter



def generate_batches(text, batch_size, sequence_length):

	block_length = len(text) // batch_size
	batches = []

	for i in range(0, block_length, sequence_length):
		batch = []
		for j in range(batch_size):
			start = j*block_length + i
			end = min(start + sequence_length, j*block_length + block_length)

			subsequence = new_data[start:end]
			subsequence_to_int = [char_to_int[char] for char in subsequence]
			batch.append(subsequence_to_int)

		batches.append(np.array(batch), dtype = int)

	return batches


def init_model(vocab, n_hidden, lr):

	seed = 0
	tf.reset_default_graph()
	tf.set_random_seed(seed = seed)

	#Hyperparameters

	hidden_units = n_hidden
	learning_rate = lr

	#Initialise tensors

	X_inp = tf.placeholder(shape = [None, None], dtype = tf.int64)
	Y_inp = tf.placeholder(shape = [None, None], dtype = tf.int64)
	#temperature = tf.placeholder(shape = [None], dtype = tf.float32) -> For bonus

	X = tf.one_hot(X_inp, depth = vocab)
	Y = tf.one_hot(Y_inp, depth = vocab)

	batch_size = tf.shape(X_inp[0])

	#Define architecture

	num_units = [hidden_units] * 2
	cells = [tf.nn.rnn_cell.LSTMCell(num_units = num_unit) for num_unit in num_units]
	stacked_rnn = tf.nn.rnn_cell.MultiRNNCell(cells)

	init_state = stacked_rnn.zero_state(batch_size, dtype = tf.float32)

	rnn_outputs, fin_state = tf.nn.dynamic_rnn(stacked_rnn, X, intial_state = init_state)


	#Shape -> (16 * 255, 256)
	rnn_outputs_flat = tf.reshape(rnn_outputs, [-1, hidden_units])

	#Shape -> (256, 48)
	W_out = tf.Variable(tf.truncated_normal(shape = (hidden_units, vocab), stddev = 0.1))

	b_out = tf.Variable(tf.zeros(shape = [vocab]))

	#Shape -> (16 * 255, 48)
	Z = tf.matmul(rnn_outputs_flat, W_out) + b_out
	Z = Z/temp

	#Shape -> (16 * 255, 48)
	Y_flat = tf.reshape(Y, [-1, vocab])

	pred = tf.nn.softmax(Z, axis = 1)

	loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels = Y_flat, logits = Z)
	loss = tf.reduce_mean(loss)

	optimizer = tf.train.AdamOptimiser(learning_rate)
	train = optimizer.minimize(loss)

	return dict(X = X_inp, Y = Y_inp, initial_state = init_state, final_state = fin_state,
				train = train, loss = loss, preds = pred, temp = temperature)

def train_and_generate(model, n_epochs, n_batches, x_batches, y_batches, temperature):

	train_losses = []
	all_sequences = []

	session = tf.Session()
	session.run(tf.global_variables_intialiser())

	for e in range(len(n_epochs)):

		loss_epoch = 0
		state = None

		for batch in range(len(n_batches)):

			X = x_batches[batch]
			Y = y_batches[batch]

			feed = {model['X'] : X, model['Y'] : Y, model['temp'] : temperature}

			if state is not None:
				feed[model['initial_state' : init_state]] = state

			l, state, _ = session.run([model['loss'], model['final_state'], model['train']], feed)

			loss_epoch += l

			if batch%100 == 0:
				print('Batch : {0}, Train loss: {1}'.format(batch, l))

		train_loss = loss_epoch/n_batches

		train_losses.append(train_loss)

		print('Epoch: {0}, Train Loss: {1}'.format(e, train_loss))



	start_text = 'and god '

	curr_state = None

	chars_to_ints = [char_to_int[char] for char in start_text]
	chars = chars_to_ints

	for i in range(n_sequences):
		for j in range(256):

			if curr_state is not None:
				feed = {model['X'] : [chars_to_ints], model['initial_state'] : curr_state}

			else:
				feed = {model['X'] : [chars_to_ints]}

			preds, curr_state = session.run([model['preds'], model['final_state']], feed)


			#Get the last predictions [this makes sense because we're only interested in what follows that last character of start_text] 
			last_pick_probs = preds[-1]

			#Pick the top 5, set all else to zero
			indices_descending = np.argsort(-last_pick_probs)


			last_pick_probs[indices_descending[5:]] = 0


			#Normalise

			last_pick_probs = last_pick_probs/np.sum(last_pick_probs)

			#Randomly choose among top 5

			chosen_char = np.random.choice(vocab, 1, p = last_pick_probs)[0]

		
			chars.append(chosen_char)


		chars = map(lambda x: int_to_char[x], chars)

		all_sequences.append(''.join(chars))


	return train_losses, all_sequences






def main():


	with open('bible.txt', encoding = 'utf-8', errors = 'ignore') as f:
		data = f.read()

	data = data.lower()

	total_chars = len(data) # -> 4432752

	to_remove = {'\n', '\\', '{', '}'}

	new_data = ''.join(char for char in data if char not in data)

	unique_chars = sorted(list(set(new_data)))

	#print(len(unique_chars)) -> 48

	#print(dict(Counter(new_data))) -> Frequency counts of unique characters

	#Map characters to integers

	char_to_int = dict((i, char) for char, i in enumerate(unique_chars))

	#Map integers to characters

	int_to_char = dict((char, i) for char, i in enumerate(unique_chars))

	batch_size = 16
	seq_length = 256


	n_epochs = 10
	temperature = 0.8

	learning_rate = 10 ** -2
	hidden_units = 512

	n_sequences = 20

	#Generate data [Drop last batch for shape match]

	batches = generate_batches(new_data, batch_size, seq_length)
	batches = batches[:-1]

	y_batches = []
	x_batches = []

	for batch in batches:
		y_batches.append(batch[:, 1:])

	for batch in batches:
		x_batches.append(batch[:, :-1])


	#Initialise model, train and generate

	model = init_model(vocab = len(unique_chars), hidden_units, learning_rate)
	losses, seq = train_and_generate(model, n_epochs, n_batches, x_batches, y_batches, temperature)


	#Plot losses

	plt.plot(losses, 'r-', label = 'Train Loss')
	plt.legend()

	plt.xlabel('Epochs')

	plt.ylabel('Loss')

	#Output generated sequences
	print(all_sequences)
