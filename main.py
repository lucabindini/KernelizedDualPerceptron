import datasets
import performances
import KernelizedDualPerceptron

from termcolor import colored

MAX_ITERATION = 15

X_train, X_validation, X_test, y_train, y_validation, y_test = datasets.load_single_dataset("banknote.csv", ',')
print(colored("--BANKNOTE DATASET--", "red"))
print(
    "X_train shape: " + str(X_train.shape) + " X_validation shape: " + str(
        X_validation.shape) + " X_test shape: " + str(
        X_test.shape))

lin_classifier = KernelizedDualPerceptron.KernelizedDualPerceptron("linear", MAX_ITERATION)
lin_classifier.train(X_train, X_validation, y_train, y_validation)
lin_prediction = lin_classifier.test(X_test)
bank_lin_accuracy = f"{performances.accuracy_rate(y_test, lin_prediction):.2f}"
print(colored("Accuracy: " + str(bank_lin_accuracy), "blue"))

pol_classifier = KernelizedDualPerceptron.KernelizedDualPerceptron("polynomial", MAX_ITERATION)
pol_classifier.train(X_train, X_validation, y_train, y_validation)
pol_prediction = pol_classifier.test(X_test)
bank_pol_accuracy = f"{performances.accuracy_rate(y_test, pol_prediction):.2f}"
print(colored("Accuracy: " + str(bank_pol_accuracy), "yellow"))

RBF_classifier = KernelizedDualPerceptron.KernelizedDualPerceptron("RBF", MAX_ITERATION)
RBF_classifier.train(X_train, X_validation, y_train, y_validation)
RBF_prediction = RBF_classifier.test(X_test)
bank_RBF_accuracy = f"{performances.accuracy_rate(y_test, RBF_prediction):.2f}"
print(colored("Accuracy: " + str(bank_RBF_accuracy), "green"))

X_train, X_validation, X_test, y_train, y_validation, y_test = datasets.load_single_dataset("biodeg.csv", ';')
print("\n\n" + colored("--BIODEGRADATION DATASET--", "red"))
print(
    "X_train shape: " + str(X_train.shape) + "X_validation shape: " + str(X_validation.shape) + "X_test shape: " + str(
        X_test.shape))

lin_classifier = KernelizedDualPerceptron.KernelizedDualPerceptron("linear", MAX_ITERATION)
lin_classifier.train(X_train, X_validation, y_train, y_validation)
lin_prediction = lin_classifier.test(X_test)
biodeg_lin_accuracy = f"{performances.accuracy_rate(y_test, lin_prediction):.2f}"
print(colored("Accuracy: " + str(biodeg_lin_accuracy), "blue"))

pol_classifier = KernelizedDualPerceptron.KernelizedDualPerceptron("polynomial", MAX_ITERATION)
pol_classifier.train(X_train, X_validation, y_train, y_validation)
pol_prediction = pol_classifier.test(X_test)
biodeg_pol_accuracy = f"{performances.accuracy_rate(y_test, pol_prediction):.2f}"
print(colored("Accuracy: " + str(biodeg_pol_accuracy), "yellow"))

RBF_classifier = KernelizedDualPerceptron.KernelizedDualPerceptron("RBF", MAX_ITERATION)
RBF_classifier.train(X_train, X_validation, y_train, y_validation)
RBF_prediction = RBF_classifier.test(X_test)
biodeg_RBF_accuracy = f"{performances.accuracy_rate(y_test, RBF_prediction):.2f}"
print(colored("Accuracy: " + str(biodeg_RBF_accuracy), "green"))

X_train, X_validation, X_test, y_train, y_validation, y_test = datasets.load_single_dataset("androgen.csv", ';')
print("\n\n" + colored("--ANDROGEN DATASET--", "red"))
print(
    "X_train shape: " + str(X_train.shape) + "X_validation shape: " + str(X_validation.shape) + "X_test shape: " + str(
        X_test.shape))

lin_classifier = KernelizedDualPerceptron.KernelizedDualPerceptron("linear", MAX_ITERATION)
lin_classifier.train(X_train, X_validation, y_train, y_validation)
lin_prediction = lin_classifier.test(X_test)
androgen_lin_accuracy = f"{performances.accuracy_rate(y_test, lin_prediction):.2f}"
print(colored("Accuracy: " + str(androgen_lin_accuracy), "blue"))

pol_classifier = KernelizedDualPerceptron.KernelizedDualPerceptron("polynomial", MAX_ITERATION)
pol_classifier.train(X_train, X_validation, y_train, y_validation)
pol_prediction = pol_classifier.test(X_test)
androgen_pol_accuracy = f"{performances.accuracy_rate(y_test, pol_prediction):.2f}"
print(colored("Accuracy: " + str(androgen_pol_accuracy), "yellow"))

RBF_classifier = KernelizedDualPerceptron.KernelizedDualPerceptron("RBF", MAX_ITERATION)
RBF_classifier.train(X_train, X_validation, y_train, y_validation)
RBF_prediction = RBF_classifier.test(X_test)
androgen_RBF_accuracy = f"{performances.accuracy_rate(y_test, RBF_prediction):.2f}"
print(colored("Accuracy: " + str(androgen_RBF_accuracy), "green"))