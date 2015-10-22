import numpy as np
import math

from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn import metrics


#Clasificador para Iris
class Classifier:
	def __init__ (self):
		self.setosa = np.empty((0,4))#Conjunto de 
		self.versicolor = np.empty((0,4))
		self.virginica = np.empty((0,4))

		self.setosa_mean = np.empty((0,4))
		self.versicolor_mean = np.empty((0,4))
		self.virginica_mean = np.empty((0,4))

	
		self.setosa_s = np.empty((0,4))
		self.versicolor_s = np.empty((0,4))
		self.virginica_s = np.empty((0,4))
		




	def mean(self, attributes_by_class):
		sum_sepal_l = 0
		sum_sepal_w = 0
		sum_petal_l = 0
		sum_petal_w = 0

		number_instances = len(attributes_by_class)

		for i in range(0, number_instances):
			sum_sepal_l += attributes_by_class[i][0]
			sum_sepal_w += attributes_by_class[i][1]
			sum_petal_l += attributes_by_class[i][2]
			sum_petal_w += attributes_by_class[i][3]
		
		mean_sepal_l = sum_sepal_l/number_instances
		mean_sepal_w =  sum_sepal_w/number_instances
		mean_petal_l =  sum_petal_l/number_instances
		mean_petal_w =  sum_petal_w/number_instances
		
		return np.array([mean_sepal_l, mean_sepal_w, mean_petal_l, mean_petal_w])

	def standard_deviation(self, attributes_by_class, mean):
		sigma_sepal_l = 0
		sigma_sepal_w = 0
		sigma_petal_l = 0
		sigma_petal_w = 0

		number_instances = len(attributes_by_class)

		for i in range(0, number_instances):
			sigma_sepal_l += (attributes_by_class[i][0] - mean[0]) * (attributes_by_class[i][0] - mean[0])		
			sigma_sepal_w += (attributes_by_class[i][1] - mean[1]) * (attributes_by_class[i][1] - mean[1])		
			sigma_petal_l += (attributes_by_class[i][2] - mean[2]) * (attributes_by_class[i][2] - mean[2])		
			sigma_petal_w += (attributes_by_class[i][3] - mean[3]) * (attributes_by_class[i][3] - mean[3])		
		
		sigma_sepal_l /= number_instances
		sigma_sepal_w /= number_instances
		sigma_petal_l /= number_instances
		sigma_petal_w /= number_instances

		sigma_sepal_l = math.sqrt(sigma_sepal_l)
		sigma_sepal_w = math.sqrt(sigma_sepal_w)
		sigma_petal_l = math.sqrt(sigma_petal_l)
		sigma_petal_w = math.sqrt(sigma_petal_w)
	
		
		return np.array([sigma_sepal_l, sigma_sepal_w, sigma_petal_l, sigma_petal_w])
 

		

	def fit(self, X_train, y_train):
		assert len(X_train) == len(y_train)

		for i in range(0, len(y_train)):

			if y_train[i] == 0:
				self.setosa = np.append(self.setosa, [X_train[i]], axis = 0)

			elif y_train[i] == 1:	
				self.versicolor = np.append(self.versicolor, [X_train[i]], axis = 0)
			else:	
				self.virginica = np.append(self.virginica, [X_train[i]], axis = 0)

	
		self.setosa_mean = self.mean(self.setosa)
		self.versicolor_mean = self.mean(self.versicolor)
		self.virginica_mean = self.mean(self.virginica)

		self.setosa_s = self.standard_deviation(self.setosa,  self.setosa_mean)
		self.versicolor_s = self.standard_deviation(self.versicolor,  self.versicolor_mean)
		self.virginica_s = self.standard_deviation(self.virginica,  self.virginica_mean)





	def aux(self, mean, sigma, value):
		return (value >= (mean - sigma) and value <= (mean + sigma))

	def predict(self, X_test):
		result  =  np.empty((0,1))
		#print(len(X_test))
		#for i in range(0, len(X_test)):
		decision = self.decision_function(X_test)

		#	result = np.append(result, np.where(decision == max(decision))[0])
		#return result
		return np.where(decision == max(decision))[0]

	def decision_function(self, X_test):
	
		sepal_l = X_test[0]
		sepal_w = X_test[1]
		petal_l = X_test[2]
		petal_w = X_test[3]

		votes_setosa = 0
		votes_versicolor= 0
		votes_virginica = 0

		if(self.aux(self.setosa_mean[0], self.setosa_s[0], sepal_l
)):
			votes_setosa = votes_setosa+1
		if(self.aux(self.setosa_mean[1], self.setosa_s[1], sepal_w
)):
			votes_setosa = votes_setosa+1
		if(self.aux(self.setosa_mean[2], self.setosa_s[2], petal_l
)):
			votes_setosa = votes_setosa+1 
		if(self.aux(self.setosa_mean[3], self.setosa_s[3], petal_w
)):
			votes_setosa = votes_setosa+1
		if(self.aux(self.versicolor_mean[0], self.versicolor_s[0], sepal_l
)):
			votes_versicolor = votes_versicolor+1
		if(self.aux(self.versicolor_mean[1], self.versicolor_s[1], sepal_w
)):
			votes_versicolor = votes_versicolor+1

		if(self.aux(self.versicolor_mean[2], self.versicolor_s[2], petal_l
)):
			votes_versicolor = votes_versicolor+1

		if(self.aux(self.versicolor_mean[3], self.versicolor_s[3], petal_w
)):
			votes_versicolor = votes_versicolor+1

		if(self.aux(self.virginica_mean[0], self.virginica_s[0], sepal_l
)):
			votes_virginica = votes_virginica+1
		if(self.aux(self.virginica_mean[1], self.virginica_s[1], sepal_w
)):
			votes_virginica = votes_virginica+1

		if(self.aux(self.virginica_mean[2], self.virginica_s[2], petal_l
)):
			votes_virginica = votes_virginica+1

		if(self.aux(self.virginica_mean[3], self.virginica_s[3], petal_w
)):
			votes_virginica = votes_virginica+1

		return np.array([votes_setosa, votes_versicolor, votes_virginica])


iris = datasets.load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)#Generamos conjunto de entrenamiento aleatoriamente
c = Classifier()
c.fit(X_train, y_train)

y_train_pred = c.predict([1,2,3,4])
print y_train_pred
#print y_train_pred
#print metrics.accuracy_score(y_train, y_train_pred)
