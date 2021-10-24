import numpy
import pandas
import sklearn
import sklearn.ensemble

import Cleaner

randomForest_max_depth = 15
randomForest_n_estimators = 80

def run():
	Cleaner.Clean('b765dc3d8076-trainset.csv')
	Cleaner.Clean('b765dc3d8076-testset_for_participants.csv')

	trainset = pandas.read_csv('b765dc3d8076-trainset_cleaned.csv')
	testset = pandas.read_csv('b765dc3d8076-testset_for_participants_cleaned.csv')
	Submission = pandas.DataFrame()

	# adding dataset_id to Submission and removing it from testset
	Submission['dataset_id'] = testset['dataset_id']
	testset.drop(columns = ['dataset_id'], inplace = True)

	predictions = Predict(Algorithm(trainset), testset)
	Submission['FRAUD_NONFRAUD'] = predictions

	Submission.to_csv('b765dc3d8076-results.csv' , index=False )



# given a cleaned trainset (pandas.DataFrame() ), return a RandomForestClassifier that can be applied to the testset
def Algorithm(trainset):
	rf = sklearn.ensemble.RandomForestClassifier(max_depth = randomForest_max_depth, n_estimators = randomForest_n_estimators)
	# rf.fit(feature variables, labels)
	rf.fit(trainset.drop(columns = ['FRAUD_NONFRAUD'] ), trainset['FRAUD_NONFRAUD'] )
	return rf


# given a trained RandomForestClassifier and a testset (pandas.DataFrame()) return a column of the predictions
def Predict(rfClassifier, testset):
	return rfClassifier.predict(testset)

run()