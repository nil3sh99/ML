from sklearn import metrics

y_true = [3,3,2,1,2]

y_pred = [2,2,2,3,3]

print(metrics.mean_absolute_error(y_true, y_pred))
print(metrics.mean_squared_error(y_true, y_pred))
