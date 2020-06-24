def accuracy(y_test, prediction):
    count = 0
    for i in range(len(y_test)):
        if y_test.iloc[i] == prediction[i]:
            count += 1.0
    return count / float(len(y_test)) * 100
