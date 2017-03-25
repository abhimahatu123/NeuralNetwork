import numpy as np
import sys

class NeuralNetwork():
	def __init__(self, inputLayerSize, hiddenLayerSize1, hiddenLayerSize2, outputLayerSize):
		#define hyperparameters
		self.inputLayerSize = inputLayerSize
		self.hiddenLayerSize1 = hiddenLayerSize1
		self.hiddenLayerSize2 = hiddenLayerSize2
		self.outputLayerSize = outputLayerSize

		#initialize weights
		self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize1)
		self.W2 = np.random.randn(self.hiddenLayerSize1, self.hiddenLayerSize2)
		self.W3 = np.random.randn(self.hiddenLayerSize2, self.outputLayerSize)

	def sigmoidDerivative(self, z):
		return np.exp(-z)/((1+np.exp(-z))**2)

	def sigmoid(self, z):
		return 1/(1 + np.exp(-z))

	def forward(self, X):
		#forward propagation through the network
		self.z2 = np.dot(X, self.W1)
		self.a2 = self.sigmoid(self.z2)

		self.z3 = np.dot(self.a2, self.W2)
		self.a3 = self.sigmoid(self.z3)

		self.z4 = np.dot(self.a3, self.W3)
		_y = self.sigmoid(self.z4)

		return _y

	def minimizeCostFunction(self, X, y):
		#compute derivatives w.r.t to individual weights
		self._y = self.forward(X)

		delta4 = np.multiply(-(y - self._y), self.sigmoidDerivative(self.z4))
		dJdW3 = np.dot(self.a3.T, delta4)

		delta3 = np.dot(delta4, self.W3.T)*self.sigmoidDerivative(self.z3)
		dJdW2 = np.dot(self.a2.T, delta3)

		delta2 = np.dot(delta3, self.W2.T)*self.sigmoidDerivative(self.z2)
		dJdW1 = np.dot(X.T, delta2)

		return dJdW1, dJdW2, dJdW3

	def train(self, X, y, numOfIterations):
		for i in xrange(numOfIterations):
			dJdW1, dJdW2, dJdW3 = neural_network.minimizeCostFunction(X, y)
			self.W1 -= dJdW1
			self.W2 -= dJdW2
			self.W3 -= dJdW3

if __name__ == "__main__":
	training_set_inputs = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
	training_set_outputs = np.array([[0, 1, 1, 0]]).T

	test_set_inputs = np.array([1, 0, 0])

	training_set_input_rows, training_set_input_cols = np.shape(training_set_inputs)
	training_set_output_rows, training_set_output_cols = np.shape(training_set_outputs)

	#initialize the neural network
	neural_network = NeuralNetwork(training_set_input_cols, 4, 6, training_set_output_cols)
	# _y = neural_network.forward(training_set_inputs)
	neural_network.train(training_set_inputs, training_set_outputs, 10000)
	print neural_network.forward(test_set_inputs)