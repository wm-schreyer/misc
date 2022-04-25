import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

''' Builds a multi-layer perceptron class and classifies handwritten digit data from 
    the MNIST database. '''

class mlp:
    def __init__(self, hnodes, momentum, learnr, epochs):
        self.hnodes = hnodes
        self.momentum = momentum
        self.learnr = learnr
        self.epochs = epochs
        self.trainAccuracy = np.array([])
        self.testAccuracy = np.array([])

    def train(self, inputsTrain, targetsTrain, inputsTest, targetsTest):
        # dataset dims
        inrows = np.shape(inputsTrain)[0]
        incols = np.shape(inputsTrain)[1]
        outcols = np.shape(targetsTrain)[1]
        # add bias to input
        inputsTrain = np.concatenate((inputsTrain, np.ones((inrows, 1))), axis=1)
        # init weights
        w1 = np.random.rand(incols+1, self.hnodes) * 0.1 - 0.05
        w2 = np.random.rand(self.hnodes+1, outcols) * 0.1 - 0.05
        # init diffs
        diff1 = np.zeros((np.shape(w1)))
        diff2 = np.zeros((np.shape(w2)))
        # core algorithm
        for epoch in range(1, self.epochs+1):
            if epoch == 1:
                # calc initial accuracy
                initial_out, _ = fwd_prop(inputsTrain, w1, w2)
                accuracy = calc_accuracy(initial_out, targetsTrain)
                self.trainAccuracy = np.append(self.trainAccuracy, accuracy)
                self.testAccuracy = np.append(self.testAccuracy, self.test(inputsTest, targetsTest, w1, w2))
                print(f'Initial training accuracy: {accuracy:.2f}')
            counter = 0
            shuff = np.arange(inrows)
            np.random.shuffle(shuff)
            inputsTrain = inputsTrain[shuff, :]
            targetsTrain = targetsTrain[shuff, :]
            for row in range(inrows):
                invec = np.array([inputsTrain[row]])
                targetvec = np.array([targetsTrain[row]])
                # fwd prop
                output, hNodes = fwd_prop(invec, w1, w2)
                if np.array_equal(output, targetvec):
                    counter += 1
                else:
                    # calc error
                    deltaOut = output * (1 - output) * (targetvec - output)
                    deltaHid = hNodes * (1 - hNodes) * (np.dot(deltaOut, np.transpose(w2)))
                    # update weights
                    diff2 = self.learnr * (np.dot(np.transpose(hNodes), deltaOut)) + (self.momentum * diff2)
                    w2 += diff2
                    diff1 = self.learnr * (np.dot(np.transpose(invec), deltaHid[:, :-1])) + (self.momentum * diff1)
                    w1 += diff1
            # calc epoch accuracy after updates
            accuracy = counter / inrows * 100
            self.trainAccuracy = np.append(self.trainAccuracy, accuracy)
            self.testAccuracy = np.append(self.testAccuracy, self.test(inputsTest, targetsTest, w1, w2))
            print(f'\nEpoch {epoch} accuracy:')
            print(f'training: {accuracy:.2f}')
        return w1, w2

    def test(self, inputsTest, targetsTest, w1, w2):
        inrows = np.shape(inputsTest)[0]
        # add bias
        inputsTest = np.concatenate((inputsTest, np.ones((inrows, 1))), axis = 1)
        counter = 0
        for row in range(inrows):
            invec = np.array([inputsTest[row]])
            targetvec = np.array([targetsTest[row]])
            # fwd prop
            output, hNodes = fwd_prop(invec, w1, w2)
            if np.array_equal(output, targetvec):
                counter += 1
        accuracy = counter / inrows * 100
        self.testAccuracy = np.append(self.testAccuracy, accuracy)
        print(f'Test accuracy:')
        print(f'{accuracy:.2f}')
        return accuracy

def encode(inn, out):
    for row in range(np.shape(inn)[0]):
        index = np.argmax(inn[row])
        out[row, index] = 0.9
    return out

def sigmoid(input):
    return 1 / (1 + np.exp(-input))

def calc_accuracy(output, target):
    rows = output.shape[0]
    counter = 0
    for row in range(rows):
        if np.argmax(output[row]) == np.argmax(target[row]):
            counter += 1
    accuracy = counter / rows * 100
    return accuracy

def fwd_prop(inputs, w1, w2):
    dot1 = np.dot(inputs, w1)
    hNodes = sigmoid(dot1)
    # add bias to hidden
    hNodes = np.concatenate((hNodes, np.ones((np.shape(hNodes)[0], 1))), axis=1)
    # calc output activations
    dot2 = np.dot(hNodes, w2)
    outputs = sigmoid(dot2)
    # convert outputs to encoded matrix
    newout = np.full(outputs.shape, 0.1, dtype=float)
    newout = encode(outputs, newout)
    return newout, hNodes

def build_cm(inputs, targets, w1, w2):
    inrows = np.shape(inputs)[0]
    outcols = np.shape(targets)[1]
    # add bias
    inputs = np.concatenate((inputs, np.ones((inrows, 1))), axis=1)
    # get output values
    output, _ = fwd_prop(inputs, w1, w2)
    outputs = np.argmax(output, 1)
    targets = np.argmax(targets, 1)
    # build confusion matrix
    confusionMatrix = np.zeros((outcols, outcols), dtype = int)
    for i in range(outcols):
        for j in range(outcols):
            confusionMatrix[i, j] = np.sum(
                np.where(targets == i, 1, 0) * np.where(outputs == j, 1, 0)
            )
    confusionFrame = pd.DataFrame(confusionMatrix,
                               columns = ['Predicted: 0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
                               index = ['Actual: 0','1','2','3','4','5','6','7','8','9'])
    return confusionFrame

# load mnist data
dfTrain = pd.read_csv('mnist_train.csv')
dfTest = pd.read_csv('mnist_test.csv')
trainTargetList = dfTrain['label'].to_numpy()
testTargetList = dfTest['label'].to_numpy()
inputsTrain = (dfTrain.drop(columns=['label']) / 255).to_numpy()
inputsTest = (dfTest.drop(columns=['label']) / 255).to_numpy()

# init encoded target lists
targetsTrain = np.full((trainTargetList.size, 10), 0.1, dtype=float)
for i in range(trainTargetList.size):
    targetsTrain[i, trainTargetList[i]] = 0.9
targetsTest = np.full((testTargetList.size, 10), 0.1, dtype=float)
for i in range(testTargetList.size):
    targetsTest[i, testTargetList[i]] = 0.9

# take subset of training data
trainSubset1 = dfTrain.sample(30000, replace=False)
subset1TargetList = trainSubset1['label'].to_numpy()
inputsSubset1 = (trainSubset1.drop(columns=['label']) / 255).to_numpy()
targetsSubset1 = np.full((subset1TargetList.size, 10), 0.1, dtype=float)
for i in range(subset1TargetList.size):
    targetsSubset1[i, subset1TargetList[i]] = 0.9

trainSubset2 = dfTrain.sample(15000, replace=False)
subset2TargetList = trainSubset2['label'].to_numpy()
inputsSubset2 = (trainSubset2.drop(columns=['label']) / 255).to_numpy()
targetsSubset2 = np.full((subset2TargetList.size, 10), 0.1, dtype=float)
for i in range(subset2TargetList.size):
    targetsSubset2[i, subset2TargetList[i]] = 0.9

# check subset sizes
print('100%:', inputsTrain.shape)
print('50%:', inputsSubset1.shape)
print('25%:', inputsSubset2.shape)

# vary hidden nodes
nodes20 = mlp(hnodes=20, momentum=0.9, learnr=0.1, epochs=50)
w1, w2 = nodes20.train(inputsTrain, targetsTrain, inputsTest, targetsTest)
nodes20.test(inputsTest, targetsTest, w1, w2)
nodes20_cm = build_cm(inputsTest, targetsTest, w1, w2)
print('20 Hidden Nodes CM')
print(nodes20_cm)
nodesplot20 = plt.figure()
plt.plot(nodes20.trainAccuracy, 'y--', marker='.', label='20 hidden nodes - training')
plt.plot(nodes20.testAccuracy, 'y', marker='.', label='20 hidden nodes - test')
plt.xlabel('epoch')
plt.ylabel('accuracy(%)')
plt.title('Accuracy as # Hidden Nodes Varies')
nodesplot20.legend(loc='upper left')
plt.show(block=True)

nodes50 = mlp(hnodes=50, momentum=0.9, learnr=0.1, epochs=50)
w1, w2 = nodes50.train(inputsTrain, targetsTrain, inputsTest, targetsTest)
nodes50.test(inputsTest, targetsTest, w1, w2)
nodes50_cm = build_cm(inputsTest, targetsTest, w1, w2)
print('50 Hidden Nodes CM')
print(nodes50_cm)
nodesplot50 = plt.figure()
plt.plot(nodes50.trainAccuracy, 'r--', marker='.', label='50 hidden nodes - training')
plt.plot(nodes50.testAccuracy, 'r', marker='.', label='50 hidden nodes - test')
plt.xlabel('epoch')
plt.ylabel('accuracy(%)')
plt.title('Accuracy as # Hidden Nodes Varies')
nodesplot50.legend(loc='upper left')
plt.show()

nodes100 = mlp(hnodes=100, momentum=0.9, learnr=0.1, epochs=50)
w1, w2 = nodes100.train(inputsTrain, targetsTrain, inputsTest, targetsTest)
nodes100.test(inputsTest, targetsTest, w1, w2)
nodes100_cm = build_cm(inputsTest, targetsTest, w1, w2)
print('100 Hidden Nodes CM')
print(nodes100_cm)
nodesplot100 = plt.figure()
plt.plot(nodes100.trainAccuracy, 'b--', marker='.', label='100 hidden nodes - training')
plt.plot(nodes100.testAccuracy, 'b', marker='.', label='100 hidden nodes - test')
plt.xlabel('epoch')
plt.ylabel('accuracy(%)')
plt.title('Accuracy as # Hidden Nodes Varies')
nodesplot100.legend(loc='upper left')
plt.show()

# vary momentum values
mome0 = mlp(hnodes=100, momentum=0, learnr=0.1, epochs=50)
w1, w2 = mome0.train(inputsTrain, targetsTrain, inputsTest, targetsTest)
mome0.test(inputsTest, targetsTest, w1, w2)
mome0_cm = build_cm(inputsTest, targetsTest, w1, w2)
print('0 Momentum CM')
print(mome0_cm)
momeplot0 = plt.figure()
plt.plot(mome0.trainAccuracy, 'y--', marker='.', label='0 momentum - training')
plt.plot(mome0.testAccuracy, 'y', marker='.', label='0 momentum - test')
plt.xlabel('epoch')
plt.ylabel('accuracy(%)')
plt.title('Accuracy as Momentum Varies')
momeplot0.legend(loc='upper left')
plt.show()

mome025 = mlp(hnodes=100, momentum=0.25, learnr=0.1, epochs=50)
w1, w2 = mome025.train(inputsTrain, targetsTrain, inputsTest, targetsTest)
mome025.test(inputsTest, targetsTest, w1, w2)
mome025_cm = build_cm(inputsTest, targetsTest, w1, w2)
print('0.25 Momentum CM')
print(mome025_cm)
momeplot025 = plt.figure()
plt.plot(mome025.trainAccuracy, 'r--', marker='.', label='0.25 momentum - training')
plt.plot(mome025.testAccuracy, 'r', marker='.', label='0.25 momentum - test')
plt.xlabel('epoch')
plt.ylabel('accuracy(%)')
plt.title('Accuracy as Momentum Varies')
momeplot025.legend(loc='upper left')
plt.show()

mome05 = mlp(hnodes=100, momentum=0.5, learnr=0.1, epochs=50)
w1, w2 = mome05.train(inputsTrain, targetsTrain, inputsTest, targetsTest)
mome05.test(inputsTest, targetsTest, w1, w2)
mome05_cm = build_cm(inputsTest, targetsTest, w1, w2)
print('0.5 Momentum CM')
print(mome05_cm)
momeplot05 = plt.figure()
plt.plot(mome05.trainAccuracy, 'b--', marker='.', label='0.5 momentum - training')
plt.plot(mome05.testAccuracy, 'b', marker='.', label='0.5 momentum - test')
plt.xlabel('epoch')
plt.ylabel('accuracy(%)')
plt.title('Accuracy as Momentum Varies')
momeplot05.legend(loc='upper left')
plt.show()

# vary the amount of training data used
sub1 = mlp(hnodes=100, momentum=0.9, learnr=0.1, epochs=50)
w1, w2 = sub1.train(inputsSubset1, targetsSubset1, inputsTest, targetsTest)
sub1.test(inputsTest, targetsTest, w1, w2)
sub1_cm = build_cm(inputsTest, targetsTest, w1, w2)
print('50% Training Data')
print(sub1_cm)
subplot1 = plt.figure()
plt.plot(sub1.trainAccuracy, 'y--', marker='.', label='50% Data - training')
plt.plot(sub1.testAccuracy, 'y', marker='.', label='50% Data - test')
plt.xlabel('epoch')
plt.ylabel('accuracy(%)')
plt.title('Accuracy at 50% Training Data')
subplot1.legend(loc='upper left')
plt.show(block=True)

sub2 = mlp(hnodes=100, momentum=0.9, learnr=0.1, epochs=50)
w1, w2 = sub2.train(inputsSubset2, targetsSubset2, inputsTest, targetsTest)
sub2.test(inputsTest, targetsTest, w1, w2)
sub2_cm = build_cm(inputsTest, targetsTest, w1, w2)
print('25% Training Data')
print(sub2_cm)
subplot2 = plt.figure()
plt.plot(sub2.trainAccuracy, 'b--', marker='.', label='25% Data - training')
plt.plot(sub2.testAccuracy, 'b', marker='.', label='25% Data - test')
plt.xlabel('epoch')
plt.ylabel('accuracy(%)')
plt.title('Accuracy at 25% Training Data')
subplot2.legend(loc='upper left')
plt.show(block=True)