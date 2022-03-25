import sys

from range_image import range_image
from feature_extraction import Feature_Extractor
from svm_classifier import SVM_Classifier

pcd_path = sys.argv[1]

ri = range_image(filename=pcd_path)
svm_model = SVM_Classifier(model_path= "./models/model1.pkl")
fe = Feature_Extractor()
 
def get_performance_value(self, model, x_value):
    
    y_value = model.predict(x_value)
    
    proba = model.predict_proba(x_value)

    return (y_value, proba)


x_value = fe.get_features(ri.output_path)
_class, proba = get_performance_value(svm_model, x_value)
print(_class, proba)