# Kick off by importing libraries, and outlining the Iris dataset
import pandas as pd   #https://github.com/jorgesleonel/Multilayer-Perceptron/blob/master/Basic%20Multi-Layer%20Perceptron.ipynb
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler  
from sklearn.neural_network import MLPClassifier 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import time

url = "/Users/jordan/Desktop/Machine Learning/winequality-white.csv"

# Let's start by naming the features
names = ["fixed acidity", "volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide",\
    "total sulfur dioxide","density","pH","sulphates","alcohol","quality"]
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
dataset = pd.read_csv(url, names=names, sep=';').iloc[1:,:]
print(dataset.iloc[1:,:].shape)
# Takes first 4 columns and assign them to variable "X"
X = dataset.iloc[:,:-1]
# Takes first 5th columns and assign them to variable "Y". Object dtype refers to strings.
y = dataset.iloc[:, -1].values
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
                iters=1800+(count*400)
                if(len(iteration_size)<27):
                    iteration_size.append(iters)
            else:
                iters=5000

            if(extra_count==2):
                if(count==0):
                    neuron_sizes = (5,5,5)
                    main_neuron_sizes.append(neuron_sizes)
                if(count==1):
                    neuron_sizes = (5,5,10)
                    main_neuron_sizes.append(neuron_sizes)
                if(count==2):
                    neuron_sizes = (5,5,15)
                    main_neuron_sizes.append(neuron_sizes)
                if(count==3):
                    neuron_sizes = (5,10,5)
                    main_neuron_sizes.append(neuron_sizes)
                if(count==4):
                    neuron_sizes = (5,10,10)
                    main_neuron_sizes.append(neuron_sizes)
                if(count==5):
                    neuron_sizes = (5,10,15)
                    main_neuron_sizes.append(neuron_sizes)
                if(count==6):
                    neuron_sizes = (5,15,5) 
                    main_neuron_sizes.append(neuron_sizes)
                if(count==7):
                    neuron_sizes = (5,15,10)
                    main_neuron_sizes.append(neuron_sizes)
                if(count==8):
                    neuron_sizes = (5,15,15)
                    main_neuron_sizes.append(neuron_sizes)
                if(count==9):
                    neuron_sizes = (10,5,5)
                    main_neuron_sizes.append(neuron_sizes)
                if(count==10):
                    neuron_sizes = (10,5,10)
                    main_neuron_sizes.append(neuron_sizes)
                if(count==11):
                    neuron_sizes = (10,5,15)
                    main_neuron_sizes.append(neuron_sizes)
                if(count==12):
                    neuron_sizes = (10,10,5)
                    main_neuron_sizes.append(neuron_sizes)
                if(count==13):
                    neuron_sizes = (10,10,10)
                    main_neuron_sizes.append(neuron_sizes)
                if(count==14):
                    neuron_sizes = (10,10,15)
                    main_neuron_sizes.append(neuron_sizes)
                if(count==15):
                    neuron_sizes = (10,15,5)
                    main_neuron_sizes.append(neuron_sizes)
                if(count==16):
                    neuron_sizes = (10,15,10)
                    main_neuron_sizes.append(neuron_sizes)
                if(count==17):
                    neuron_sizes = (10,15,15)
                    main_neuron_sizes.append(neuron_sizes)
                if(count==18):
                    neuron_sizes = (15,5,5)
                    main_neuron_sizes.append(neuron_sizes)
                if(count==19):
                    neuron_sizes = (15,5,10)
                    main_neuron_sizes.append(neuron_sizes)
                if(count==20):
                    neuron_sizes = (15,5,15)
                    main_neuron_sizes.append(neuron_sizes)
                if(count==21):
                    neuron_sizes = (15,10,5)
                    main_neuron_sizes.append(neuron_sizes)
                if(count==22):
                    neuron_sizes = (15,10,10)
                    main_neuron_sizes.append(neuron_sizes)
                if(count==23):
                    neuron_sizes = (15,10,15)
                    main_neuron_sizes.append(neuron_sizes)
                if(count==24):
                    neuron_sizes = (15,15,5)
                    main_neuron_sizes.append(neuron_sizes)
                if(count==25):
                    neuron_sizes = (15,15,10)
                    main_neuron_sizes.append(neuron_sizes)
                if(count==26):
                    neuron_sizes = (15,15,15)
                    main_neuron_sizes.append(neuron_sizes)
                
            else:
                neuron_sizes = (10,10,10)




            mlp = MLPClassifier(hidden_layer_sizes=neuron_sizes, max_iter=iters)  
            mlp.fit(X_train, y_train)#.values.ravel())

            in_predictions = mlp.predict(X_train)
            out_predictions = mlp.predict(X_test)

            

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
            main_neuron_in_accuracy.append(neuron_in_accuracy)
            main_neuron_out_accuracy.append(neuron_out_accuracy)
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
plt.savefig("test_size_vs_accuracy_nn.png")
plt.clf()

plt.title("Iterations vs runtime")
plt.xlabel("Iterations")
plt.ylabel("Runtime")
plt.plot(final_iterations, avg_times, 'o-', color="b",label="Runtime")
plt.legend(loc="best")
plt.savefig("iterations_vs_time_nn.png")

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



"""
import os
import sys

train = True
test = False
if len(sys.argv) >= 2 and sys.argv[1] == 'test' :
	test = True
	train = False
	os.remove('/Users/jordan/Desktop/Machine Learning/output.txt')

if test: data = open("/Users/jordan/Desktop/Machine Learning/testingData.txt", "r")
else: data = open("/Users/jordan/Desktop/Machine Learning/trainingData.txt", "r")

with open("/Users/jordan/Desktop/Machine Learning/constants.txt", "r") as constants:
	num_nodes = int(constants.readline()[:-1])
	bias_factor = float(constants.readline()[:-1])
	learn_rate = float(constants.readline()[:-1])

weight = []
inputNodes = []

try:
	with open("/Users/jordan/Desktop/Machine Learning/weights.txt", "r") as weights:
		inputWeight = weights.readline()
		while inputWeight:
			weight.append(float(inputWeight[:-1]))
			inputWeight = weights.readline()
except IOError:
    print("error writing weights")
    for i in range(num_nodes):
        weight.append(1.0)

inputLine = data.readline()

print(inputLine)

while inputLine: 
    inputLine = inputLine[:-1].split(",")
    print(inputLine[-1])
    if(inputLine[-1] == 'setosa'):
        inputLine[-1] = 10.0
        print("setosa")
    if(inputLine[-1] == 'versicolor'):
        inputLine[-1] = 20.0
        print("vers")
    if(inputLine[-1] == 'virginica'):
        inputLine[-1] = 30.0
        print("virg")
    if test: inputNodes = inputLine
    else:
        inputNodes = inputLine[:-1]
        outputNode = inputLine[-1]
    print(inputLine)
    # Weighted Sum function
    wSum = -bias_factor
    print("input nodes and weight array")
    print(inputNodes)
    #print(type(weight))
    for i in range(num_nodes):
        wSum += float(inputNodes[i]) * weight[i]
    print("wSum")
    print(wSum)
    # Activation function
    if wSum >= 0.0 : prediction = 20.0
    else : prediction = 10.0

    if test:
        # Writing prediction to output file
        with open("/Users/jordan/Desktop/Machine Learning/output.txt", "a") as output:
            for i in range(num_nodes):
                output.write(str(inputNodes[i]) + ' ')
            output.write(str(prediction) + '\n')

    if train:
        # Weight update rule
        # Stochastic gradient descent
        print(type(inputNodes[0]))
        print(type(outputNode-prediction))
        for i in range(num_nodes):
            weight[i] += learn_rate * ((outputNode) - prediction) * (float(inputNodes[i]))

    inputLine = data.readline()

if train:
	# Writing updated weights to weights.txt
	with open("/Users/jordan/Desktop/Machine Learning/weights.txt", "w") as weights:
		for i in range(num_nodes):
			weights.write(str(weight[i])+'\n')

data.close()
"""







"""
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd


def step_function(x):
    if x<0:
        return 0
    else:
        return 1


training_set = [((0, 0, 0), 0), ((0, 0, 1), 1), ((1, 0, 0), 1), ((1, 1, 1), 1)]

# ploting data points using seaborn (Seaborn requires dataframe)
plt.figure(0)

x1 = [training_set[i][0][0] for i in range(4)]
x2 = [training_set[i][0][1] for i in range(4)]
x3 = [training_set[i][0][2] for i in range(4)]

y = [training_set[i][1] for i in range(4)]

df = pd.DataFrame(
    {'x1': x1,
     'x2': x2,
     'x3': x3,
     'y': y
    })
    
sns.lmplot("x1", "x2", data=df, hue='y', fit_reg=False, markers=["o", "s"])


# parameter initialization
w = np.random.rand(3)
errors = [] 
eta = .5
epoch = 30
b = 0


# Learning
for i in range(epoch):
    for x, y in training_set:
      # u = np.dot(x , w) +b
        u = sum(x*w) + b
        
        error = y - step_function(u) 
      
        errors.append(error) 
        for index, value in enumerate(x):
            #print(w[index])
            w[index] += eta * error * value
            b += eta*error
   
        ''' produce all decision boundaries
            a = [0,-b/w[1]]
            c = [-b/w[0],0]
            plt.figure(1)
            plt.plot(a,c)
        '''
            
            
# final decision boundary
a = [0,-b/w[1]]
c = [-b/w[0],0]
plt.plot(a,c)
plt.savefig("test1.png")
   
# ploting errors   
plt.figure(2)
plt.ylim([-1,1]) 
plt.plot(errors)
plt.savefig("test2.png")

"""
