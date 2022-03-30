import sys

from range_image import range_image
from feature_extraction import Feature_Extractor
from svm_classifier import SVM_Classifier

pcd_path = sys.argv[1]

ri = range_image(filename=pcd_path)


#####################################################################
# clf = SVM_Classifier(model_path= "./models/model1.pkl")
# print("model loaded...")
# fe = Feature_Extractor()
# print("feature extractor created...")
 
# def get_performance_value(model, x_value):
    
#     y_value = model.predict(x_value.reshape(1, -1))
    
#     proba = model.predict_proba(x_value.reshape(1, -1))

#     return (int(y_value), float(max(proba[0])))


# x_value = fe.get_features(ri.output_path)
# print("features extracted...")
# _class, proba = get_performance_value(clf.svm_model, x_value)
# print(_class, proba)
#####################################################################



#####################################################################
fe = Feature_Extractor()
clf = SVM_Classifier(orb = True)
_, orb_features = fe.get_orb_features(fe.get_cv_image(ri.output_path))
print(orb_features.shape)
_class, proba = clf.match_orb_features(orb_features) 

print(_class, proba)

#####################################################################