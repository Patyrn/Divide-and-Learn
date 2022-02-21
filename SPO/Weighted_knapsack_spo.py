import os

from SPO.SPO_dp_lr import *

def weighted_knapsack_SPO(train_set, test_set, n_iter=10, capacities = [30], file_name_suffix='empty', dest_folder = "Tests/icon/Easy/kfolds/spo/"):
	dir_path = os.path.dirname(os.path.abspath(__file__))


	filename = dest_folder + file_name_suffix
	filename = os.path.join(dir_path, filename)


	X_1gtrain = train_set.get('X')
	y_train = train_set.get('Y')
	X_1gtest = test_set.get('X')
	y_test = test_set.get('Y')

	benchmarks_weights_train = train_set.get('benchmarks_weights')

	X_1gvalidation = X_1gtrain[0:2880, :]
	y_validation = y_train[0:2880]

	y_train = y_train[2880:]
	X_1gtrain = X_1gtrain[2880:, :]


	weights = benchmarks_weights_train[0]

	lrs = [1e-2]
	for lr in lrs:
		for repitition  in range(n_iter):
			for capa in capacities:
				print(capa)
				file_name_full = filename + '_c' + str(capa) + '.csv'
				clf = SGD_SPO_dp_lr([capa],weights,lr=lr,verbose=True,epochs=3,use_relaxation=True,validation_relax= False, store_result=True)
				pdf = clf.fit(X_1gtrain,y_train,X_1gtest,y_test,X_1gvalidation,y_validation)
				pdf['capacity'] = capa
				with open(file_name_full, 'a+') as f:
					pdf.to_csv(f, mode='a', header=f.tell() == 0, index=False)
