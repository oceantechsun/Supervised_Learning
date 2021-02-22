# Load libraries
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn import datasets
# Import train_test_split function
from sklearn.model_selection import train_test_split
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler  
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/cmc/cmc.data"

# Let's start by naming the features
names = ["wife's age", "wife's education","husband's education","number of children","wife's religion","wife employed",\
    "husband's occupation","standard of living","media exposure","contraceptive used"]

dataset = pd.read_csv(url, header = None, names= names)

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
est_size = []


"""
#Encoding the categorical column
dataset = dataset.replace({"class":  {"Iris-setosa":1,"Iris-versicolor":2, "Iris-virginica":3}})
#Visualize the new dataset
dataset.head()

plt.figure(1)
sns.heatmap(dataset.corr())
plt.title('Correlation On iris Classes')
"""


X = dataset.iloc[:,:-1]
y = dataset.iloc[:, -1].values
#print(y.shape[0])



for extra_count in range(3):
    for outer_count in range(2):
        in_accuracy_ratings = []
        out_accuracy_ratings = []
        neuron_in_accuracy = []
        neuron_out_accuracy = []
        execution_times = []
        for count in range(27):
            if(extra_count==1):
                #print("start time:")
                
                start = time.time()
                #print(start)
            # Reading the dataset through a Pandas function
            ## Split data into training and testing sets.

            if(extra_count==0):
                t_size=0.90-(float(count)*.03)
            else:
                t_size=.30

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=t_size)

            if(len(test_sizes)<27):
                test_sizes.append(X_train.shape[0])
            
            scaler = StandardScaler()  
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)  
            X_test = scaler.transform(X_test)

            if(extra_count==1):
                ls=10+(count*4)
                if(len(est_size)<27):
                    est_size.append(ls)
            else:
                ls=15

            if(extra_count==2):
                lr=1.0-(count*.033)
                if(len(main_neuron_sizes)<27):
                    #neuron_sizes = lr
                    main_neuron_sizes.append(lr)
            else:
                lr=0.3

            ada = AdaBoostClassifier(n_estimators=ls,learning_rate=lr)
            
            ## Fit the model on the training data.
            ada.fit(X_train, y_train)
            ## See how the model performs on the test data.
            in_predictions = ada.predict(X_train)
            out_predictions = ada.predict(X_test)

            if(extra_count==0):
                in_accuracy_ratings.append(metrics.accuracy_score(in_predictions,y_train))
                out_accuracy_ratings.append(metrics.accuracy_score(out_predictions,y_test))
                #rmse_in.append(mean_squared_error(y_train,in_predictions))
                #rmse_out.append(mean_squared_error(y_test,out_predictions))
            elif(extra_count==1):
                #print("end time: ")
                end = time.time()
                #print(end-start)
                execution_times.append(end-start)
            else:
                neuron_in_accuracy.append(metrics.accuracy_score(y_train,in_predictions))
                neuron_out_accuracy.append(metrics.accuracy_score(y_test,out_predictions))


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
            #main_neuron_in_accuracy.append(neuron_in_accuracy)
            #main_neuron_out_accuracy.append(neuron_out_accuracy)
            break

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
        final_iterations = np.array(est_size)
    else:
        pass


plt.title("Learning Curves")
plt.xlabel("Training examples")
plt.ylabel("Score")
plt.grid()
plt.fill_between(final_test_sizes, avg_in_accuracy - std_in_accuracy, avg_in_accuracy + std_in_accuracy, alpha=0.1, color="g")
plt.plot(final_test_sizes, avg_in_accuracy, 'o-', color="g",label="Training score")
plt.fill_between(final_test_sizes, avg_out_accuracy - std_out_accuracy, avg_out_accuracy + std_out_accuracy, alpha=0.1, color="r")
plt.plot(final_test_sizes, avg_out_accuracy, 'o-', color="r",label="Cross-validation score")
plt.legend(loc="best")
plt.savefig("test_size_vs_accuracy_ada2.png")
plt.clf()

plt.title("Learning rate and accuracy score")
plt.xlabel("Learning rate")
plt.ylabel("Score")
plt.grid()
#plt.fill_between(final_test_sizes, avg_in_accuracy - std_in_accuracy, avg_in_accuracy + std_in_accuracy, alpha=0.1, color="g")
plt.plot(main_neuron_sizes, neuron_in_accuracy, 'o-', color="g",label="Training score")
#plt.fill_between(final_test_sizes, avg_out_accuracy - std_out_accuracy, avg_out_accuracy + std_out_accuracy, alpha=0.1, color="r")
plt.plot(main_neuron_sizes, neuron_out_accuracy, 'o-', color="r",label="Cross-validation score")
plt.legend(loc="best")
plt.savefig("learning_rate_vs_accuracy_ada2.png")
plt.clf()

plt.title("Number of Estimators vs runtime")
plt.xlabel("Estimators")
plt.ylabel("Runtime")
plt.plot(final_iterations, avg_times, 'o-', color="b",label="Runtime")
plt.legend(loc="best")
plt.savefig("iterations_vs_time_ada2.png")

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
print(neuron_in_accuracy)
print("main_neuron_out_accuracy")
print(neuron_out_accuracy)