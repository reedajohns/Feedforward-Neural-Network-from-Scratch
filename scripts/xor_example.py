# USAGE
# python xor_example.py

# import packages
from fundamentals.nn import NeuralNet
import numpy as np

# construct the XOR dataset
X = np.array([[0, 0],
			  [1, 1],
			  [1, 0],
			  [0, 1]])
y = np.array([[0],
			  [1],
			  [1],
			  [0]])

# define a 2-2-1 neural network
nn = NeuralNet([2, 2, 1], alpha=0.5)
# Train the neural net!
nn.fit(X, y, epochs=20000)

# Lets loop the XOR data points and send them through the trained nn
for (x, target) in zip(X, y):
	# Make a prediction for the point
	prediction = nn.predict(x)
	prediction = prediction[0][0]
	# Step function
	step = 1 if prediction > 0.5 else 0
	# Print some info
	print("[INFO] data={}, ground-truth={}, pred={:.4f}, step={}".format(x, target[0], prediction, step))