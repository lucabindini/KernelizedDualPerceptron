def accuracy_rate(y_test, prediction):
    count = 0
    y_test = y_test.to_numpy()
    for i in range(len(y_test)):
        if y_test[i] == prediction[i]:
            count += 1
    return count / float(len(y_test)) * 100


def error_rate(y_test, prediction):
    return 1 - (accuracy_rate(y_test, prediction) / 100)
