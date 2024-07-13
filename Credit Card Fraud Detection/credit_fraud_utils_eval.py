from sklearn.metrics import f1_score,accuracy_score,classification_report
import pickle

def model_perdict(model,x):
    y = model.predict(x)
    return y 

def eval(y_label, y_perdict):
    return classification_report(y_label,y_perdict)

def model_eval(models,x,y):
    rue = []
    idx =0
    for key, classifier in models.items():
        print(f'---------------------------{classifier}---------')
        y_predict = model_perdict(classifier,x)
        rue.append(eval(y,y_predict))
        print(rue[idx])
        idx+=1
        
    return rue

def save(model):
    with open('model.pkl', 'wb') as file:
        pickle.dump(model, file)


