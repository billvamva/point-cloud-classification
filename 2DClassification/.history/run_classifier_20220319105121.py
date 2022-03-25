    
def get_performance_value(self, model, x_value):
    
    y_value = model.predict(x_value)
    
    proba = model.predict_proba(x_value)

    return (y_value, proba)



_class, proba = get_performance_value(svm_model, x_value)
print(_class, proba)