import datasets as dt
import evaluate_accuracy as ea
import KernelizedDualPerceptron
from termcolor import colored

X_train, X_test, y_train, y_test = dt.load_single_dataset("banknote.csv", ',')
print(colored("--BANKNOTE DATASET--", "red"))
max_iteration = 15

lin_classifier = KernelizedDualPerceptron.KernelizedDualPerceptron("linear", max_iteration)
lin_classifier.train(X_train, y_train)
lin_prediction = lin_classifier.test(X_test)
# print(y_test)
print(lin_prediction)
print(colored(str(ea.accuracy(y_test, lin_prediction)), "blue"))

pol_classifier = KernelizedDualPerceptron.KernelizedDualPerceptron("polynomial", max_iteration)
pol_classifier.train(X_train, y_train)
pol_prediction = pol_classifier.test(X_test)
print(pol_prediction)
print(colored(str(ea.accuracy(y_test, pol_prediction)), "yellow"))

RBF_classifier = KernelizedDualPerceptron.KernelizedDualPerceptron("RBF", max_iteration)
RBF_classifier.train(X_train, y_train)
RBF_prediction = RBF_classifier.test(X_test)
print(RBF_prediction)
print(colored(str(ea.accuracy(y_test, RBF_prediction)), "green"))

X_train, X_test, y_train, y_test = dt.load_single_dataset("biodeg.csv", ';')
print("\n\n" + colored("--BIODEG DATASET--", "red"))

lin_classifier = KernelizedDualPerceptron.KernelizedDualPerceptron("linear", max_iteration)
lin_classifier.train(X_train, y_train)
lin_prediction = lin_classifier.test(X_test)
# print(y_test)
print(lin_prediction)
print(colored(str(ea.accuracy(y_test, lin_prediction)), "blue"))

pol_classifier = KernelizedDualPerceptron.KernelizedDualPerceptron("polynomial", max_iteration)
pol_classifier.train(X_train, y_train)
pol_prediction = pol_classifier.test(X_test)
print(pol_prediction)
print(colored(str(ea.accuracy(y_test, pol_prediction)), "yellow"))

RBF_classifier = KernelizedDualPerceptron.KernelizedDualPerceptron("RBF", max_iteration)
RBF_classifier.train(X_train, y_train)
RBF_prediction = RBF_classifier.test(X_test)
print(RBF_prediction)
print(colored(str(ea.accuracy(y_test, RBF_prediction)), "green"))

X_train, X_test, y_train, y_test = dt.load_single_dataset("androgen.csv", ';')
print("\n\n" + colored("--ANDROGEN DATASET--", "red"))

lin_classifier = KernelizedDualPerceptron.KernelizedDualPerceptron("linear", max_iteration)
lin_classifier.train(X_train, y_train)
lin_prediction = lin_classifier.test(X_test)
# print(y_test)
print(lin_prediction)
print(colored(str(ea.accuracy(y_test, lin_prediction)), "blue"))

pol_classifier = KernelizedDualPerceptron.KernelizedDualPerceptron("polynomial", max_iteration)
pol_classifier.train(X_train, y_train)
pol_prediction = pol_classifier.test(X_test)
print(pol_prediction)
print(colored(str(ea.accuracy(y_test, pol_prediction)), "yellow"))

RBF_classifier = KernelizedDualPerceptron.KernelizedDualPerceptron("RBF", max_iteration)
RBF_classifier.train(X_train, y_train)
RBF_prediction = RBF_classifier.test(X_test)
print(RBF_prediction)
print(colored(str(ea.accuracy(y_test, RBF_prediction)), "green"))
