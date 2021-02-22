import numpy as np    #https://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler  
import time


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/cmc/cmc.data"

# Let's start by naming the features
names = ["wife's age", "wife's education","husband's education","number of children","wife's religion","wife employed",\
    "husband's occupation","standard of living","media exposure","contraceptive used"]

dataset = pd.read_csv(url, header = None, names= names)

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
            if(extra_count==0):
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
                ls=5+(count*2)
                if(len(iteration_size)<27):
                    iteration_size.append(ls)
            else:
                ls=20

            

            if(extra_count==2):
                neuron_sizes = count+2
                if(len(main_neuron_sizes)<27):
                    main_neuron_sizes.append(neuron_sizes)
                
            else:
                neuron_sizes = 5


            knn = KNeighborsClassifier(n_neighbors=neuron_sizes, leaf_size=ls)


            
            knn.fit(X_train, y_train)#.values.ravel())

            in_predictions = knn.predict(X_train)
            out_predictions = knn.predict(X_test)

            

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
                neuron_in_accuracy.append(accuracy_score(y_train,in_predictions))
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
plt.savefig("test_size_vs_accuracy_knn2.png")
plt.clf()

plt.title("Number of Neighbors and accuracy score")
plt.xlabel("Number of Neighbors")
plt.ylabel("Score")
plt.grid()
#plt.fill_between(final_test_sizes, avg_in_accuracy - std_in_accuracy, avg_in_accuracy + std_in_accuracy, alpha=0.1, color="g")
plt.plot(main_neuron_sizes, neuron_in_accuracy, 'o-', color="g",label="Training score")
#plt.fill_between(final_test_sizes, avg_out_accuracy - std_out_accuracy, avg_out_accuracy + std_out_accuracy, alpha=0.1, color="r")
plt.plot(main_neuron_sizes, neuron_out_accuracy, 'o-', color="r",label="Cross-validation score")
plt.legend(loc="best")
plt.savefig("num_neighbors_vs_accuracy_knn2.png")
plt.clf()

plt.title("Leaf size vs runtime")
plt.xlabel("Leaf Size")
plt.ylabel("Runtime")
plt.plot(final_iterations, avg_times, 'o-', color="b",label="Runtime")
plt.legend(loc="best")
plt.savefig("leaf_size_vs_time_knn2.png")

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
# Now transforming categorial into numerical values



"""
iris = iris.replace({"class":  {"Iris-setosa":1,"Iris-versicolor":2, "Iris-virginica":3}})

for extra_count in range(2):
    for outer_count in range(5):
        accuracy_ratings = []
        execution_times = []
        for count in range(10):
            if(extra_count==1):
                #print("start time:")
                
                start = time.time()
                #print(start)
            # Reading the dataset through a Pandas function
            ## Split data into training and testing sets.

            if(extra_count==0):
                t_size=0.90-(float(count)*.088)
            else:
                t_size=.30

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=t_size)

            if(len(test_sizes)<10):
                test_sizes.append(X_train.shape[0])
            
            scaler = StandardScaler()  
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)  
            X_test = scaler.transform(X_test)

            if(extra_count==1):
                ls=5+(count*5)
                if(len(leaf_size)<10):
                    leaf_size.append(ls)
            else:
                ls=30

            knn = KNeighborsClassifier(n_neighbors=5, leaf_size=ls)

            ## Fit the model on the training data.
            knn.fit(X_train, y_train)
            ## See how the model performs on the test data.
            y_pred = knn.predict(X_test)

            #knn.score(X_test, y_test)

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
        else:
            main_times.append(execution_times)


    if(extra_count==0):   #maybe should remove this, just think mean should be taken after secondmost loop finishes
        final_accuracy = np.array(main_accuracy_ratings)
        avg_accuracy = np.mean(final_accuracy, axis=0)
        final_test_sizes = np.array(test_sizes)
    else:
        final_times = np.array(main_times)
        avg_times = np.mean(final_times, axis=0)
        final_leaf_size = np.array(leaf_size)
plt.plot(final_test_sizes, avg_accuracy)
plt.savefig("iris_test_size_vs_accuracy_knn.png")
plt.clf()
plt.plot(final_leaf_size, avg_times)
plt.savefig("iris_leaf_size_vs_time_knn.png")

"""