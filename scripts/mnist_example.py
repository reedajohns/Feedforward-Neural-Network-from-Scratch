# USAGE
# python mnist_example.py

# import packages
from fundamentals.nn import NeuralNet
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets

# load the MNIST dataset and apply min/max scaling to scale the
# pixel intensity values to the range [0, 1]
# (Where each image is represented by an 8 x 8 = 64-dim feature vector)
print("[INFO] loading MNIST (sample) dataset...")
digits = datasets.load_digits()
# Convert to float
data = digits.data.astype("float")
data = (data - data.min()) / (data.max() - data.min())
print("[INFO] samples: {}, dim: {}".format(data.shape[0],
	data.shape[1]))

# Split the dataset
(trainX, testX, trainY, testY) = train_test_split(data,
	digits.target, test_size=0.25)

# convert the labels from integers to vectors (one hot encoding)
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

# train the network
print("[INFO] training network...")
nn = NeuralNet([trainX.shape[1], 32, 16, 10])
# Train the network
nn.fit(trainX, trainY, epochs=800)

# evaluate the network
print("[INFO] evaluating network...")
predictions = nn.predict(testX)
predictions = predictions.argmax(axis=1)
print(classification_report(testY.argmax(axis=1), predictions))