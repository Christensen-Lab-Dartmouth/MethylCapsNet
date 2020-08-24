#conda install -c rapidsai -c h2oai -c conda-forge h2o4gpu-cuda92 cuml
import fire # cuml , h2o4gpu
# conda install -c h2oai -c conda-forge h2o4gpu-cuda10
from pymethylprocess.MethylationDataTypes import MethylationArray
# import cudf
import numpy as np
from dask.diagnostics import ProgressBar
# from cuml.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, classification_report
import pickle
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.ensemble import StackingClassifier
import torch, copy
import torch.nn as nn
import torch_scatter
# from skorch import NeuralNetClassifier
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
# from skorch.callbacks import EpochScoring
from methylcapsnet.build_capsules import return_final_capsules
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, Pipeline
import pysnooper
from mlxtend.classifier import StackingClassifier
import dask
from scipy import sparse
from sklearn.base import clone
from pathos.multiprocessing import ProcessPool
import torch.optim as optim
import torchsnooper
from itertools import combinations
from scipy.stats import pearsonr
# from multiprocessing import set_start_method
from torch.multiprocessing import Pool, Process, set_start_method

try:
	set_start_method('spawn')
except RuntimeError:
	pass
# # from cuml.linear_model import MBSGDClassifier as LogisticRegression
#
# # methylnet-torque run_torque_job -c "python comparison_methods.py" -gpu -a "conda activate methylnet_lite" -ao "-A Brock -l nodes=1:ppn=1" -q gpuq -t 5 -n 1 # -sup
# nohup python group_logistic_regression.py group_lasso --l1_vals [0.,0.1,1.,5.,10.,20.,100.,1000.] --n_epochs 50 &

def beta2M(beta):
	beta[beta==0.]=beta[beta==0.]+1e-8
	beta[beta==1.]=beta[beta==1.]-1e-8
	return np.log2(beta/(1.-beta))

class ParallelStackingClassifier(StackingClassifier):
	def __init__(self, classifiers, meta_classifier,
				 use_probas=False, drop_last_proba=False,
				 average_probas=False, verbose=0,
				 use_features_in_secondary=False,
				 store_train_meta_features=False,
				 use_clones=True,n_jobs=20):
		super(ParallelStackingClassifier,self).__init__(classifiers,
												meta_classifier,
												use_probas,
												drop_last_proba,
												average_probas,
												verbose,
												use_features_in_secondary,
												store_train_meta_features,
												use_clones)
		self.n_jobs=n_jobs

	def fit(self, X, y, sample_weight=None, capsules=[]):
		""" Fit ensemble classifers and the meta-classifier.

		Parameters
		----------
		X : {array-like, sparse matrix}, shape = [n_samples, n_features]
			Training vectors, where n_samples is the number of samples and
			n_features is the number of features.
		y : array-like, shape = [n_samples] or [n_samples, n_outputs]
			Target values.
		sample_weight : array-like, shape = [n_samples], optional
			Sample weights passed as sample_weights to each regressor
			in the regressors list as well as the meta_regressor.
			Raises error if some regressor does not support
			sample_weight in the fit() method.

		Returns
		-------
		self : object

		"""
		if self.use_clones:
			self.clfs_ = clone(self.classifiers)
			self.meta_clf_ = clone(self.meta_classifier)
		else:
			self.clfs_ = self.classifiers
			self.meta_clf_ = self.meta_classifier

		if self.verbose > 0:
			print("Fitting %d classifiers..." % (len(self.classifiers)))

		clfs=[]
		#print(self.clfs_)
		for i,clf in enumerate(self.clfs_):
			if sample_weight is None:
				clfs.append(dask.delayed(lambda capsule:clf.fit(X,y))(capsules[i]))
			else:
				clfs.append(dask.delayed(lambda capsule:clf.fit(X,y,sample_weight=sample_weight))(capsules[i]))

		with ProgressBar():
			self.clfs_=dask.compute(*clfs,scheduler='threading',num_workers=self.n_jobs)

		self.classifiers=self.clfs_

		meta_features = self.predict_meta_features(X)

		if self.store_train_meta_features:
			self.train_meta_features_ = meta_features

		if not self.use_features_in_secondary:
			pass
		elif sparse.issparse(X):
			meta_features = sparse.hstack((X, meta_features))
		else:
			meta_features = np.hstack((X, meta_features))

		if sample_weight is None:
			self.meta_clf_.fit(meta_features, y)
		else:
			self.meta_clf_.fit(meta_features, y, sample_weight=sample_weight)

		return self

class CapsuleSelection(TransformerMixin, BaseEstimator):
	def __init__(self, capsule=[],name=''):
		self.capsule=capsule
		self.name=name

	def fit(self, X, y=None, **fit_params):
		return self

	def transform(self, X, y=None, **fit_params):
		caps_X=X.loc[:,self.capsule].values#cudf.from_pandas(X.loc[:,self.capsules])
		return caps_X

	def fit_transform(self, X, y=None, **fit_params):
		return self.fit(X).transform(X)

	def get_params(self, deep=True):
		"""
		:param deep: ignored, as suggested by scikit learn's documentation
		:return: dict containing each parameter from the model as name and its current value
		"""
		return {}

	def set_params(self, **parameters):
		"""
		set all parameters for current objects
		:param parameters: dict containing its keys and values to be initialised
		:return: self
		"""
		for parameter, value in parameters.items():
			setattr(self, parameter, value)
		return self

class CapsuleLogReg(BaseEstimator, ClassifierMixin):
	def __init__(self, capsule=[],name='', model=LogisticRegression()):
		self.capsule=capsule
		self.name=name
		self.model=model

	def fit(self, X, y=None, **fit_params):
		self.model.fit(X.loc[:,self.capsule],y)
		return self

	def predict(self, X, y=None, **fit_params):
		return self.model.predict(X.loc[:,self.capsule])#cudf.from_pandas(X.loc[:,self.capsules])

	def predict_proba(self, X, y=None, **fit_params):
		return self.model.predict_proba(X.loc[:,self.capsule])#cudf.from_pandas(X.loc[:,self.capsules])

	def fit_predict(self, X, y=None, **fit_params):
		return self.fit(X,y).predict(X)

	def get_params(self, deep=True):
		"""
		:param deep: ignored, as suggested by scikit learn's documentation
		:return: dict containing each parameter from the model as name and its current value
		"""
		return {}

	def set_params(self, **parameters):
		"""
		set all parameters for current objects
		:param parameters: dict containing its keys and values to be initialised
		:return: self
		"""
		for parameter, value in parameters.items():
			setattr(self, parameter, value)
		return self

# class LogisticRegressionModel(torch.nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(LogisticRegressionModel, self).__init__()
#         self.linear = torch.nn.Linear(input_dim, output_dim)
#
#     def forward(self, x):
#         outputs = self.linear(x)
#         return F.softmax(outputs, dim=-1)
#
# class MyClassifier(NeuralNetClassifier):
#     def get_loss(self, y_pred, y_true, X, training=False):
#         return self.criterion_(y_pred, y_true)

def fit_logreg(train_methyl_array='train_val_test_sets/train_methyl_array.pkl',
			   val_methyl_array='train_val_test_sets/val_methyl_array.pkl',
			   test_methyl_array='train_val_test_sets/test_methyl_array.pkl',
			   l1_vals=np.hstack((np.arange(0.01,1.1,0.1),np.array([10.,20.,50.,100.]))),
			   outcome_col='disease_only',
			   min_capsule_len=5,
			   capsule_choice=['gene'],
			   n_jobs=20
			   ):

	datasets=dict(train=train_methyl_array,
				  val=val_methyl_array,
				  test=test_methyl_array)

	# LogisticRegression = lambda ne, lr: net = NeuralNetClassifier(LogisticRegressionModel,max_epochs=ne,lr=lr,iterator_train__shuffle=True, callbacks=[EpochScoring(LASSO)])

	X=dict()
	Y=dict()
	le=LabelEncoder()
	for k in ['train','val','test']:
		datasets[k]=MethylationArray.from_pickle(datasets[k])
		X[k]=datasets[k].beta#cudf.from_pandas(datasets[k].beta)#
		X[k].loc[:,:]=beta2M(X[k].loc[:,:].values)
		Y[k]=le.fit_transform(datasets[k].pheno[outcome_col]) if k=='train' else le.transform(datasets[k].pheno[outcome_col])#cudf.Series(, dtype = np.float32 )

	capsules,_,names=return_final_capsules(datasets['train'], capsule_choice, min_capsule_len, None,None, 0, '', '')
	# make_pipeline(CapsuleSelection(capsule,name), LogisticRegression(penalty='l1', C=1./l1,class_weight='balanced'))
	# capsules=capsules[:2]#[capsule for capsule in capsules]
	# names=names[:2]#[name for name in names]
	build_stacking_model=lambda l1: ParallelStackingClassifier(n_jobs=n_jobs,meta_classifier=LogisticRegression(penalty='l1', n_jobs=n_jobs, C=1./l1,class_weight='balanced', solver='saga'),use_clones=False,classifiers=[make_pipeline(CapsuleSelection(capsule,name),LogisticRegression(penalty='l1', C=1./l1,class_weight='balanced',solver='saga')) for capsule,name in zip(capsules,names) if len(capsule)])

	def get_score(l1,capsules):
		print('Fitting l1: {}'.format(l1))
		score=f1_score(build_stacking_model(l1).fit(X['train'],Y['train'],capsules=capsules).predict(X['val']),Y['val'],average='macro')
		return l1,score

	scores=[get_score(l1,capsules) for l1 in l1_vals]
	# pool=ProcessPool(nodes=8)
	# scores=pool.map(lambda l1: get_score(l1,capsules), l1_vals)
	# for l1 in l1_vals:
	# 	scores.append(get_score(l1,capsules))#scores.append(dask.delayed(get_score)(l1))
	# scores.append((l1,f1_score(reg.predict(X['val']).to_pandas().values.flatten().astype(int),Y['val'].to_pandas().values.flatten().astype(int),average='macro')))
	scores=np.array(scores)#dask.compute(*scores,scheduler='processes')
	np.save('l1_scores.npy',scores)
	l1=scores[np.argmin(scores[:,1]),0]
	reg = build_stacking_model(l1)#LogisticRegression(penalty='l1', C=1./l1)#LogisticRegression(ne,lr)#
	reg.fit(X['train'],Y['train'],capsules=capsules)
	print(classification_report(le.inverse_transform(Y['test']),le.inverse_transform(reg.predict(X['test']))))
	# print(classification_report(le.inverse_transform(Y['test'].to_pandas().values.flatten().astype(int)),le.inverse_transform(reg.predict(X['test']).to_pandas().values.flatten().astype(int))))
	pickle.dump(dict(model=reg,features=names),open('stacked_model.pkl','wb'))
	#pickle.dump(dict(coef=reg.coef_.T,index=datasets['test'].beta.columns.values,columns=le.classes_),open('logreg_coef_.pkl','wb'))
	#pd.DataFrame(reg.coef_.T,index=datasets['test'].beta.columns.values,columns=le.classes_).to_csv('logreg_coef.pkl')

class GroupLasso(nn.Module):
	def __init__(self, n_features, n_classes, names, cpg_arr, capsule_sizes, l1):
		super().__init__()
		self.n_features=n_features
		self.n_classes=n_classes
		self.names=names
		self.cpg_arr=cpg_arr
		self.capsule_sizes=capsule_sizes
		self.l1=l1
		self.model=nn.Linear(self.n_features,self.n_classes)
		if torch.cuda.is_available():
			self.model=self.model.cuda()
		self.weights=[]
		self.groups=[]
		for i,name in enumerate(self.names):
			idx=self.cpg_arr.loc[self.cpg_arr['feature']==name,'cpg'].values
			w=self.model.weight[:,idx].flatten()#**2
			self.weights.append(w)
			self.groups.extend([i]*len(w))
		self.weights=torch.cat(self.weights,0)
		# self.groups=torch.tensor(np.array(self.groups),device=('cuda' if torch.cuda.is_available() else 'cpu'))
		# self.sqrt_group_sizes=torch.tensor(np.sqrt(np.array(self.capsule_sizes)),device=('cuda' if torch.cuda.is_available() else 'cpu'))

	def forward(self,x):
		return self.model(x)

	def penalize(self, lasso_for_loop=True):
		lasso_for_loop=True
		if lasso_for_loop:
			W=self.model.weight
			group_lasso_penalty=[]
			for i,name in enumerate(self.names):
				idx=self.cpg_arr.loc[self.cpg_arr['feature']==name,'cpg'].values
				group_lasso_penalty.append(np.sqrt(self.capsule_sizes[i])*torch.norm(W[:,idx].flatten(),2))
			group_lasso_penalty=sum(group_lasso_penalty)
		else:
			group_lasso_penalty=self.l1*torch.sum(torch.sqrt(torch_scatter.scatter_add(self.weights**2,self.groups))*self.sqrt_group_sizes)#sum(group_lasso_penalty)
		# print(group_lasso_penalty)
		return group_lasso_penalty

	def return_weights(self):
		W=self.model.weight
		return np.array([np.sqrt(self.capsule_sizes[i])**(-1)*torch.norm(W[:,self.cpg_arr.loc[self.cpg_arr['feature']==self.names[i],'cpg'].values].flatten(),2).detach().cpu().numpy() for i in range(len(self.capsule_sizes))])

def fit_group_lasso(train_methyl_array='train_val_test_sets/train_methyl_array.pkl',
			   val_methyl_array='train_val_test_sets/val_methyl_array.pkl',
			   test_methyl_array='train_val_test_sets/test_methyl_array.pkl',
			   l1_vals=np.hstack((np.arange(0.01,1.1,0.1),np.array([10.,20.,50.,100.]))).tolist(),
			   outcome_col='disease_only',
			   min_capsule_len=5,
			   capsule_choice=['gene'],
			   n_jobs=0,
			   n_epochs=10,
			   output_results_pickle='group_lasso_model.pkl',
			   output_file='group_lasso_importances.csv',
			   batch_size=1280,
			   lr=0.0001,
			   predict=False
			   ):

	# if torch.cuda.is_available():
	# 	torch.set_default_tensor_type('torch.cuda.FloatTensor')

	datasets=dict(train=train_methyl_array,
				  val=val_methyl_array,
				  test=test_methyl_array)

	# LogisticRegression = lambda ne, lr: net = NeuralNetClassifier(LogisticRegressionModel,max_epochs=ne,lr=lr,iterator_train__shuffle=True, callbacks=[EpochScoring(LASSO)])

	X=dict()
	Y=dict()
	le=LabelEncoder()
	for k in ['train','val','test']:
		datasets[k]=MethylationArray.from_pickle(datasets[k])

	capsules,cpgs,names,cpg_arr=return_final_capsules(datasets['train'], capsule_choice, min_capsule_len, None,None, 0, '', '', return_original_capsule_assignments=True)

	cpgs=np.unique(cpgs)

	cpg2idx=dict(zip(cpgs,np.arange(len(cpgs))))

	cpg_arr.loc[:,'cpg']=cpg_arr.loc[:,'cpg'].map(cpg2idx)

	capsule_sizes=[len(capsule) for capsule in capsules]

	for k in ['train','val','test']:
		X[k]=datasets[k].beta.loc[:,cpgs]#cudf.from_pandas(datasets[k].beta)#
		X[k].loc[:,:]=beta2M(X[k].loc[:,:].values)
		Y[k]=le.fit_transform(datasets[k].pheno[outcome_col]) if k=='train' else le.transform(datasets[k].pheno[outcome_col])#cudf.Series(, dtype = np.float32 )

	n_classes=len(np.unique(Y['train']))
	n_cpgs=len(cpgs)

	class_weights=torch.tensor(compute_class_weight('balanced', np.unique(Y['train']), Y['train'])).float()

	if torch.cuda.is_available():
		class_weights=class_weights.cuda()

	dataloaders={}

	for k in ['train','val','test']:
		X[k]=torch.tensor(X[k].values).float()
		Y[k]=torch.tensor(Y[k]).long()
		dataloaders[k]=DataLoader(TensorDataset(X[k],Y[k]),batch_size=min(batch_size,X[k].shape[0]),shuffle=(k=='train'),num_workers=n_jobs)

	def get_res(logreg_model,dataloader=dataloaders['test'], return_pred=False):
		y_res={'pred':[],'true':[]}
		for i,(x,y) in enumerate(dataloader):
			if torch.cuda.is_available():
				x=x.cuda()
				y=y.cuda()
			y_res['true'].extend(y.detach().cpu().numpy().flatten().tolist())
			y_res['pred'].extend(logreg_model(x).argmax(1).detach().cpu().numpy().flatten().tolist())
		y_true=le.inverse_transform(np.array(y_res['true']))
		y_pred=le.inverse_transform(np.array(y_res['pred']))
		print(classification_report(y_true,y_pred))
		if return_pred:
			return (y_true,y_pred)
		return f1_score(y_true,y_pred,average='macro')

	# @torchsnooper.snoop()
	def train_model(l1, return_model=False):
		logreg_model=GroupLasso(n_cpgs, n_classes,names,cpg_arr,capsule_sizes,l1)#nn.Module(nn.Linear())#nn.Sequential(,nn.LogSoftmax())
		if torch.cuda.is_available():
			logreg_model=logreg_model.cuda()
			logreg_model.weights=logreg_model.weights.cuda()
			# logreg_model.groups=logreg_model.groups.cuda()
			# logreg_model.sqrt_group_sizes=logreg_model.sqrt_group_sizes.cuda()
		optimizer = optim.Adam(logreg_model.parameters(), lr=lr)
		# scheduler=
		criterion=nn.CrossEntropyLoss(weight=class_weights)

		for epoch in range(n_epochs):
			running_loss={'train':[],'val':[]}
			for phase in ['train','val']:
				logreg_model.train(phase=='train')
				for i,(x,y) in enumerate(dataloaders[phase]):
					if torch.cuda.is_available():
						x=x.cuda()
						y=y.cuda()
					optimizer.zero_grad()
					#print(y.shape,logreg_model(x).shape)
					y_pred=logreg_model(x)
					loss=criterion(y_pred,y)
					loss=loss+logreg_model.penalize()

					#group_lasso()#group_lasso(logreg_model[0].weight)
					if phase=='train':
						loss.backward()
						optimizer.step()
					# 	optimizer.zero_grad()
					# penalty=logreg_model.penalize()
					# if phase=='train':
					# 	penalty.backward(retain_graph=True)
					# 	optimizer.step()
					# loss=loss+penalty
					item_loss=loss.item()
					print("Epoch {}[Batch {}] - {} Loss {} ".format(epoch, i, phase, item_loss),flush=True)
					running_loss[phase].append(item_loss)
				running_loss[phase]=np.mean(running_loss[phase])
			print("Epoch {} - Train Loss {} , Val Loss {}".format(epoch, running_loss['train'], running_loss['val']),flush=True)
			if epoch and running_loss['val']<=min_val_loss:
				min_val_loss=running_loss['val']
				best_model_weights=copy.deepcopy(logreg_model.state_dict())
			elif not epoch:
				min_val_loss=running_loss['val']
		logreg_model.load_state_dict(best_model_weights)

		if not return_model:
			return l1,get_res(logreg_model,dataloader=dataloaders['val'])
		else:
			return get_res(logreg_model,dataloader=dataloaders['val'],return_pred=False),logreg_model
	# pool=ProcessPool(nodes=8)
	# l1_f1=np.array(pool.map(train_model, l1_vals))
	if len(l1_vals)>1:
		l1_f1=np.array(dask.compute(*[dask.delayed(train_model)(l1) for l1 in l1_vals],scheduler='threading'))#np.array([train_model(l1) for l1 in l1_vals])

		l1=l1_f1[np.argmax(l1_f1[:,1]),0]
	else:
		l1_f1=None
		l1=l1_vals[0]

	# group_lasso=GroupLasso(len(cpgs),len(np.unique(Y['train'])),names,cpg_arr,capsule_sizes,l1)

	f1,logreg_model=train_model(l1,return_model=True)
	logreg_model.train(False)

	y_true,y_pred=get_res(logreg_model,return_pred=True)

	torch.save(dict(model=logreg_model,l1=l1_f1, y_true=y_true, y_pred=y_pred, f1=f1),output_results_pickle)

	weights=logreg_model.return_weights()
	pd.DataFrame(dict(zip(names,weights)),index=['importances']).T.to_csv(output_file)

def get_correlation_network_(train_methyl_array='train_val_test_sets/train_methyl_array.pkl',
			   val_methyl_array='train_val_test_sets/val_methyl_array.pkl',
			   test_methyl_array='train_val_test_sets/test_methyl_array.pkl',
			   min_capsule_len=5,
			   capsule_choice=['gene'],
			   n_jobs=20,
			   output_file='corr_mat.pkl'
			   ):

	# if torch.cuda.is_available():
	# 	torch.set_default_tensor_type('torch.cuda.FloatTensor')

	datasets=dict(train=train_methyl_array,
				  val=val_methyl_array,
				  test=test_methyl_array)

	# LogisticRegression = lambda ne, lr: net = NeuralNetClassifier(LogisticRegressionModel,max_epochs=ne,lr=lr,iterator_train__shuffle=True, callbacks=[EpochScoring(LASSO)])

	X=dict()
	for k in ['train','val','test']:
		X[k]=MethylationArray.from_pickle(datasets[k]).beta

	capsules,_,names,cpg_arr=return_final_capsules(datasets['train'], capsule_choice, min_capsule_len, None,None, 0, '', '', return_original_capsule_assignments=True)

	caps={names[i]:X['train'].loc[:,cpgs].apply(np.median,axis=1).values for i,cpgs in enumerate(capsules)} # maybe add more values

	df=pd.DataFrame(1.,index=names,columns=names)
	for c1,c2 in combinations(names,r=2):
		df.loc[c1,c2]=pearsonr(caps[c1],caps[c2])

	df.to_pickle(output_file)


class Commands(object):
	def __init__(self):
		pass

	def fit_logreg(self,
						train_methyl_array='train_val_test_sets/train_methyl_array.pkl',
					   val_methyl_array='train_val_test_sets/val_methyl_array.pkl',
					   test_methyl_array='train_val_test_sets/test_methyl_array.pkl',
					   l1_vals=np.hstack((np.arange(0.01,1.1,0.1),np.array([10.,20.,50.,100.]))),
					   outcome_col='disease_only',
					   min_capsule_len=5,
					   capsule_choice=['gene'],
					   n_jobs=20):
		fit_logreg(train_methyl_array,
					val_methyl_array,
					test_methyl_array,
					l1_vals,
					outcome_col,
					min_capsule_len,
					capsule_choice,
					n_jobs)

	def group_lasso(self,
						train_methyl_array='train_val_test_sets/train_methyl_array.pkl',
					   val_methyl_array='train_val_test_sets/val_methyl_array.pkl',
					   test_methyl_array='train_val_test_sets/test_methyl_array.pkl',
					   l1_vals=np.hstack((np.arange(0.01,1.1,0.1),np.array([10.,20.,50.,100.]))).tolist(),
					   outcome_col='disease_only',
					   min_capsule_len=5,
					   capsule_choice=['gene'],
					   n_jobs=20,
					   n_epochs=200,
					   output_results_pickle='group_lasso_model.pkl',
					   output_file='group_lasso_importances.csv'):
		fit_group_lasso(train_methyl_array,
						   val_methyl_array,
						   test_methyl_array,
						   l1_vals,
						   outcome_col,
						   min_capsule_len,
						   capsule_choice,
						   n_jobs,
						   n_epochs,
						   output_results_pickle,
						   output_file)

	def get_correlation_network(train_methyl_array='train_val_test_sets/train_methyl_array.pkl',
				   val_methyl_array='train_val_test_sets/val_methyl_array.pkl',
				   test_methyl_array='train_val_test_sets/test_methyl_array.pkl',
				   min_capsule_len=5,
				   capsule_choice=['gene'],
				   n_jobs=20,
				   output_file='corr_mat.pkl'
				   ):
		get_correlation_network_(train_methyl_array,
					   val_methyl_array,
					   test_methyl_array,
					   min_capsule_len,
					   capsule_choice,
					   n_jobs,
					   output_file
					   )

def main():
	fire.Fire(Commands)

if __name__=='__main__':
	main()
