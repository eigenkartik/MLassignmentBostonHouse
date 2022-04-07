import pickle

x_predict=[[2,2,7,1,0,2,6,5,2,5,4,2]]
filename='final_model.pkl'

model=pickle.load(open(filename,'rb'))
prediction=model.predict(x_predict)

print(prediction)