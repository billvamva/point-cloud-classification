from range_image import range_image
from feature_extraction import Feature_Extractor
from svm_classifier import SVM_Classifier

 
def get_performance_value(self, model, x_value):
    
    y_value = model.predict(x_value)
    
    proba = model.predict_proba(x_value)

    return (y_value, proba)



_class, proba = get_performance_value(svm_model, x_value)
print(_class, proba)