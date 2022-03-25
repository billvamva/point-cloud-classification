    
def get_performance_value(self, model, x_value):
    
    y_value = model.predict(x_value)
    
    proba = model.predict_proba(x_value)

    return (y_value, proba)



_class, proba = self.get_performance_value(self.svm_model, self.x_value)
print(_class, proba)