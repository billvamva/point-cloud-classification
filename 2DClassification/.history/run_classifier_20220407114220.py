import sys
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from multiprocessing import Process

# from range_image import range_image
from feature_extraction import Feature_Extractor
from svm_classifier import SVM_Classifier

pcd_path = sys.argv[1]

# ri = range_image(filename=pcd_path)


#####################################################################
# clf = SVM_Classifier(model_path= "./models/model1.pkl")
# print("model loaded...")
# fe = Feature_Extractor()
# print("feature extractor created...")
 
# def get_performance_value(model, x_value):
    
#     y_value = model.predict(x_value.reshape(1, -1))
    
#     proba = model.predict_proba(x_value.reshape(1, -1))

#     return (int(y_value), float(max(proba[0])))

# print(ri.output_path)

# x_value = fe.get_features(ri.output_path)
# print("features extracted...")
# _class, proba = get_performance_value(clf.svm_model, x_value)
# print(_class, proba)
#####################################################################



#####################################################################
fe = Feature_Extractor()
clf = SVM_Classifier(orb = True)
class_dict = {"cube": '1', "cylinder": '2', "car": '3', "sphere": '4'}
path = './range_images/'
y_test = []
y_pred = []


for count, file in enumerate(os.listdir(path)):
    filename = os.fsdecode(file)
    if filename.split(".")[-1] == "png":
        print(f"{count}/{len(os.listdir(path))}, {filename}")
        y_test.append(filename.split("_")[0])
        _, orb_features = fe.get_orb_features(fe.get_cv_image(path + filename))
        _class, proba = clf.match_orb_features(orb_features) 
        y_pred.append(_class)

print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred, normalize = 'true'))


#####################################################################