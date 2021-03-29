import pandas as pd
from math import sqrt, pi, exp
import matplotlib.pyplot as plt
# from sklearn.naive_bayes import GaussianNB

#functions
def norm_pdf(Mu, sigma, x):
    return(1.0/(sigma * sqrt(2*pi))) * exp(-0.5*((x - Mu)/sigma) ** 2)

def scores(predicted, test, class1):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(len(predicted)):
        if predicted[i] == class1:
            if test[i] == class1:
                tp += 1
            else:
                fp += 1
        else:
            if test[i] == class1:
                fn += 1
            else:
                tn += 1
    precision = tp/(tp + fp)
    recall = tp/(tp + fn)
    f1 = 2*precision*recall/(precision + recall)
    accuracy = (tp + tn)/len(predicted)
    specifity = tn/(tn + fp)
    confusen_matrix = [[tp, fp],[fn, tn]]
    print("precision :", precision)
    print("recall :", recall)
    print("F1 :", f1)
    print("accuracy :", accuracy)
    print("specifity :", specifity)
    print("confusen matrix:")
    for i in confusen_matrix:
        print(i)


# importing dataset
dataset = pd.read_excel("Book1.xlsx")
testset = pd.read_excel("test_set.xlsx")
# print(dataset.head())
# print(testset.shape)
# print(testset.head())

dataset = dataset.drop('Id', axis=1)
testset = testset.iloc[:, 1:]

# print(testset.shape)

#print(dataset.shape)

#print(dataset.groupby('Species').size())
#print(dataset.info())
#print(dataset.describe())
#dataset.plot(kind='box', sharex=False, sharey=False)
#plt.show()

#data preparation
X_test = testset.iloc[:, :-1].values
Y_test = testset.iloc[:, -1].values

# print(X_test)

X = dataset.groupby("Species")
class1 = X.get_group("Iris-setosa")
class2 = X.get_group("Iris-versicolor")
Mu1 = class1.mean().to_list()
Mu2 = class2.mean().to_list()
sigma1 = class1.std().to_list()
sigma2 = class2.std().to_list()

pw1 = len(class1)/len(dataset)
pw2 = len(class2)/len(dataset)
# print(X_train.mean())
# print(X_train.std())





Y_predict = []

for item in X_test:
    PxgivenW1 = 1
    PxgivenW2 = 1
    for i, v in enumerate(item):
        PxgivenW1 = norm_pdf(Mu1[i], sigma1[i], v) * PxgivenW1
        PxgivenW2 = norm_pdf(Mu2[i], sigma2[i], v) * PxgivenW2
    PxgivenW1 = pw1 * PxgivenW1
    PxgivenW2 = pw2 * PxgivenW2
    if PxgivenW1 > PxgivenW2:
        Y_predict.append("Iris-setosa")
    else:
        Y_predict.append("Iris-versicolor")

# print(pd.DataFrame(Y_predict))
# print(pd.DataFrame(Y_test))
Y_test[2] = "Iris-versicolor"

scores(Y_predict, Y_test, "Iris-setosa")