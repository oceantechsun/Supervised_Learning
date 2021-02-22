import pandas as pd   #https://medium.com/@pinnzonandres/iris-classification-with-svm-on-python-c1b6e833522c
import matplotlib.pyplot as plt
import seaborn as sns   #https://stackoverflow.com/questions/22239691/code-for-best-fit-straight-line-of-a-scatter-plot-in-python
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from scipy.stats import pearsonr

import time
import numpy as np
def best_fit(X, Y):

    xbar = sum(X)/len(X)
    ybar = sum(Y)/len(Y)
    n = len(X) # or len(Y)

    numer = sum([xi*yi for xi,yi in zip(X, Y)]) - n * xbar * ybar
    denum = sum([xi**2 for xi in X]) - n * xbar**2

    b = numer / denum
    a = ybar - b * xbar

    print('best fit line:\ny = {:.2f} + {:.2f}x'.format(a, b))

    return a, b

url = "/Users/jordan/Desktop/Machine Learning/winequality-white.csv"

# Let's start by naming the features
names = ["fixed acidity", "volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide",\
    "total sulfur dioxide","density","pH","sulphates","alcohol","quality"]

dataset = pd.read_csv(url, names=names, sep=';').iloc[1:,:]
print(dataset.iloc[1:,:].shape)
# Takes first 4 columns and assign them to variable "X"
X = dataset.iloc[:,:-1]
# Takes first 5th columns and assign them to variable "Y". Object dtype refers to strings.
y = dataset.iloc[:, -1].values
# we only take the first two features. We could avoid this ugly
# slicing by using a two-dim dataset


main_in_accuracy_ratings = []
main_out_accuracy_ratings = []
main_neuron_sizes = []
main_neuron_in_accuracy = []
main_neuron_out_accuracy = []
#main_rmse_in= []
#main_rmse_out= []
main_times=[]
test_sizes = []
iteration_size = []
for extra_count in range(3):
    for outer_count in range(2):
        in_accuracy_ratings = []
        out_accuracy_ratings = []
        neuron_in_accuracy = []
        neuron_out_accuracy = []
        #rmse_in= []
        #rmse_out= []
        execution_times = []
        for count in range(27):
            if(extra_count==1):
                #print("start time:")
                
                start = time.time()
                #print(start)


            # y actually contains all categories or classes:
            #y.Class.unique()

            # Now transforming categorial into numerical values
            #le = preprocessing.LabelEncoder()
            #y = y.apply(le.fit_transform)


            # Now for train and test split (80% of  dataset into  training set and  other 20% into test data)
            if(extra_count==0 or extra_count==2):
                t_size=0.90-(float(count)*.03)
            else:
                t_size=.30
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = t_size)
            if(len(test_sizes)<27):
                test_sizes.append(X_train.shape[0])
            #print("xtrain size")
            #print(X_train.shape)
            # Feature scaling
            scaler = StandardScaler()  
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)  
            X_test = scaler.transform(X_test)


            # Finally for the MLP- Multilayer Perceptron
            if(extra_count==1):
                iters=280+(count*2000)
                if(len(iteration_size)<27):
                    iteration_size.append(iters)
            else:
                iters=-1

            

            if(extra_count==2 and outer_count==1):

                neuron_sizes = 'linear'
                #if(len(main_neuron_sizes)<27):
                #    main_neuron_sizes.append(neuron_sizes)
                
            else:
                neuron_sizes = 'poly'



            classifier = SVC(kernel = neuron_sizes, random_state = 0)
            #Fit the model for the data

            
            classifier.fit(X_train, y_train)#.values.ravel())

            in_predictions = classifier.predict(X_train)
            out_predictions = classifier.predict(X_test)

            

            #print(predictions)

            # Last thing: evaluation of algorithm performance in classifying flowers
            #print(confusion_matrix(y_test,predictions))  
            #print("ze accuracy senor")
            #print(accuracy_score(y_test,predictions))
            #print(type(classification_report(y_test,predictions)))
            #print(classification_report(y_test,predictions)[1,0])
            #print(classification_report(y_test,predictions))
            if(extra_count==0):
                in_accuracy_ratings.append(accuracy_score(y_train,in_predictions))
                out_accuracy_ratings.append(accuracy_score(y_test,out_predictions))
                #rmse_in.append(mean_squared_error(y_train,in_predictions))
                #rmse_out.append(mean_squared_error(y_test,out_predictions))
            elif(extra_count==1):
                #print("end time: ")
                end = time.time()
                #print(end-start)
                execution_times.append(end-start)
            else:
                if(outer_count==0):
                    neuron_in_accuracy.append(accuracy_score(y_test,out_predictions))
                else:
                    neuron_out_accuracy.append(accuracy_score(y_test,out_predictions))
        if(extra_count==0):
            main_in_accuracy_ratings.append(in_accuracy_ratings)
            main_out_accuracy_ratings.append(out_accuracy_ratings)
            #main_rmse_in.append(rmse_in)
            #main_rmse_out.append(rmse_out)
            #final_accuracy = np.array(main_accuracy_ratings)
            #avg_accuracy = np.mean(final_accuracy, axis=0)
            #final_test_sizes = np.array(test_sizes)
        elif(extra_count==1):
            main_times.append(execution_times)
            #final_times = np.array(execution_times)
            #final_iterations = np.array(iteration_size)
            break
        else:
            if(outer_count==0):
                main_neuron_in_accuracy=neuron_in_accuracy
            else:
                main_neuron_out_accuracy=neuron_out_accuracy
            #break
    if(extra_count==0):   #maybe should remove this, just think mean should be taken after secondmost loop finishes
        final_in_accuracy = np.array(main_in_accuracy_ratings)
        final_out_accuracy = np.array(main_out_accuracy_ratings)
        #final_in_rmse = np.array(main_rmse_in)
        #final_out_rmse = np.array(main_rmse_out)
        avg_in_accuracy = np.mean(final_in_accuracy, axis=0)
        avg_out_accuracy = np.mean(final_out_accuracy, axis=0)
        std_in_accuracy = np.std(final_in_accuracy, axis=0)
        std_out_accuracy = np.std(final_out_accuracy, axis=0)
        final_test_sizes = np.array(test_sizes)
    elif(extra_count==1):
        final_times = np.array(main_times)
        avg_times = np.mean(final_times, axis=0)
        final_iterations = np.array(iteration_size)
    else:
        pass

#plt.scatter(final_test_sizes, avg_accuracy)
plt.title("Learning Curves")
plt.xlabel("Training examples")
plt.ylabel("Score")
plt.grid()
plt.fill_between(final_test_sizes, avg_in_accuracy - std_in_accuracy, avg_in_accuracy + std_in_accuracy, alpha=0.1, color="g")
plt.plot(final_test_sizes, avg_in_accuracy, 'o-', color="g",label="Training score")
plt.fill_between(final_test_sizes, avg_out_accuracy - std_out_accuracy, avg_out_accuracy + std_out_accuracy, alpha=0.1, color="r")
plt.plot(final_test_sizes, avg_out_accuracy, 'o-', color="r",label="Cross-validation score")
plt.legend(loc="best")
plt.savefig("test_size_vs_accuracy_svm.png")
plt.clf()

plt.title("Linear vs Poly accuracy score")
plt.xlabel("Training examples")
plt.ylabel("Score")
plt.grid()
#plt.fill_between(final_test_sizes, avg_in_accuracy - std_in_accuracy, avg_in_accuracy + std_in_accuracy, alpha=0.1, color="g")
a, b = best_fit(final_test_sizes, main_neuron_in_accuracy)
plt.scatter(final_test_sizes, main_neuron_in_accuracy, color="b",label="Poly cross-validation score")
yfit = [a + b * xi for xi in final_test_sizes]
poly_corr = pearsonr(final_test_sizes, main_neuron_in_accuracy)
plt.plot(final_test_sizes, yfit, color="b",label="Best fit for Poly score")

#plt.fill_between(final_test_sizes, avg_out_accuracy - std_out_accuracy, avg_out_accuracy + std_out_accuracy, alpha=0.1, color="r")
c, d = best_fit(final_test_sizes, main_neuron_out_accuracy)

plt.scatter(final_test_sizes, main_neuron_out_accuracy, color="g",label="Linear cross-validation score")
zfit = [c + d * yi for yi in final_test_sizes]
linear_corr = pearsonr(final_test_sizes, main_neuron_out_accuracy)
plt.plot(final_test_sizes, zfit, color="g",label="Best fit for Linear score")

plt.legend(loc="best")
plt.savefig("linear_poly_vs_accuracy_svm.png")
plt.clf()

plt.title("Maximum Iterations vs runtime")
plt.xlabel("Maximum Iterations")
plt.ylabel("Runtime")
plt.plot(final_iterations, avg_times, 'o-', color="b",label="Runtime")
plt.legend(loc="best")
plt.savefig("leaf_size_vs_time_svm.png")

print("avg in sample accuracy")
print(avg_in_accuracy)
print("avg out ofsample accuracy")
print(avg_out_accuracy)
print("std in sample accuracy")
print(std_in_accuracy)
print("std out of sample accuracy")
print(std_out_accuracy)
print("runtimes")
print(avg_times)
print("sample sizes")
print(final_test_sizes)
#print("in sample rmse")
#print(final_in_rmse)
#print("out of sample rmse")
#print(final_out_rmse)

print("main_neuron_in_accuracy")
print(main_neuron_in_accuracy)
print("main_neuron_out_accuracy")
print(main_neuron_out_accuracy)
print("polycore")
print(poly_corr)
print("linear core")
print(linear_corr)




"""
#Define the col names
colnames=["sepal_length_in_cm", "sepal_width_in_cm","petal_length_in_cm","petal_width_in_cm", "class"]

#Read the dataset
dataset = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header = None, names= colnames )

#Data
dataset.head()

#Encoding the categorical column
dataset = dataset.replace({"class":  {"Iris-setosa":1,"Iris-versicolor":2, "Iris-virginica":3}})
#Visualize the new dataset
dataset.head()

"""

"""


X = dataset.iloc[:,:-1]
y = dataset.iloc[:, -1].values
#print(y.shape[0])



#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


#Create the SVM model

classifier = SVC(kernel = 'linear', random_state = 0)
#Fit the model for the data

classifier.fit(X_train, y_train)

#Make the prediction
y_pred = classifier.predict(X_test)




cm = confusion_matrix(y_test, y_pred)
print(cm)


accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print(accuracies)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


main_accuracy_ratings = []
main_times = []
test_sizes = []
iteration_size = []
for extra_count in range(2):
    for outer_count in range(5):
        accuracy_ratings = []
        execution_times = []
        for count in range(10):
            if(extra_count==1):
                print("start time:")
                
                start = time.time()
                print(start)


            if(extra_count==0):
                t_size=0.90-(float(count)*.088)
            else:
                t_size=.30
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=t_size, random_state = 0)
            if(len(test_sizes)<10):
                test_sizes.append(X_train.shape[0])

            scaler = StandardScaler()  
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)  
            X_test = scaler.transform(X_test)

            if(extra_count==1):
                iters=280+(count*11000)
                if(len(iteration_size)<10):
                    iteration_size.append(iters)
            else:
                iters=-1
            
            classifier = SVC(kernel = 'linear', max_iter = iters, random_state = 0)
            #Fit the model for the data

            classifier.fit(X_train, y_train)

            #Make the prediction
            y_pred = classifier.predict(X_test)

            cm = confusion_matrix(y_test, y_pred)

            if(extra_count==0):
                accuracy_ratings.append(accuracy_score(y_test,y_pred))
                #print(accuracy_score(y_test,y_pred))
            else:
                #print("end time: ")
                end = time.time()
                #print(end-start)
                execution_times.append(end-start)
        if(extra_count==0):
            main_accuracy_ratings.append(accuracy_ratings)
            #final_accuracy = np.array(main_accuracy_ratings)
            #avg_accuracy = np.mean(final_accuracy, axis=0)
            #final_test_sizes = np.array(test_sizes)
        else:
            main_times.append(execution_times)
            #final_times = np.array(execution_times)
            #final_iterations = np.array(iteration_size)
            #break
    if(extra_count==0):   #maybe should remove this, just think mean should be taken after secondmost loop finishes
        final_accuracy = np.array(main_accuracy_ratings)
        avg_accuracy = np.mean(final_accuracy, axis=0)
        final_test_sizes = np.array(test_sizes)
    else:
        final_times = np.array(main_times)
        avg_times = np.mean(final_times, axis=0)
        final_iterations = np.array(iteration_size)
plt.plot(final_test_sizes, avg_accuracy)
plt.savefig("iris_test_size_vs_accuracy_svm.png")
plt.clf()
plt.plot(final_iterations, avg_times)
plt.savefig("iris_iterations_vs_time_svm.png")

"""