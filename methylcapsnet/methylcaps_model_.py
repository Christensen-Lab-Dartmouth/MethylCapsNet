import pandas as pd
from pymethylprocess.MethylationDataTypes import MethylationArray
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")
from pybedtools import BedTool
import numpy as np
from functools import reduce
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import os
import pysnooper
import argparse
import pickle
from sklearn.metrics import classification_report
import click
import methylcapsnet
from methylcapsnet.build_capsules import *
from methylcapsnet.methylcaps_data_models import *
import sqlite3
import os
import glob
import dask
from dask.diagnostics import ProgressBar
from pathos.multiprocessing import Pool
import multiprocessing
import dask.bag as db
from distributed import Client, LocalCluster, get_task_stream
RANDOM_SEED=42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

@pysnooper.snoop('train.log')
def model_capsnet_(train_methyl_array='train_val_test_sets/train_methyl_array.pkl',
					val_methyl_array='train_val_test_sets/val_methyl_array.pkl',
					interest_col='disease',
					n_epochs=10,
					n_bins=0,
					bin_len=1000000,
					min_capsule_len=300,
					primary_caps_out_len=45,
					caps_out_len=45,
					hidden_topology='30,80,50',
					gamma=1e-2,
					decoder_topology='100,300',
					learning_rate=1e-2,
					routing_iterations=3,
					overlap=0.,
					custom_loss='none',
					gamma2=1e-2,
					job=0,
					capsule_choice=['genomic_binned'],
					custom_capsule_file='',
					test_methyl_array='',
					predict=False,
					batch_size=16,
					limited_capsule_names_file='',
					gsea_superset='',
					tissue='',
					number_sets=25,
					use_set=False,
					gene_context=False,
					select_subtypes=[],
					fit_pas=False,
					l1_l2='',
					custom_capsule_file2='',
					min_capsules=5):

	capsule_choice=list(capsule_choice)
	#custom_capsule_file=list(custom_capsule_file)
	hlt_list=filter(None,hidden_topology.split(','))
	if hlt_list:
		hidden_topology=list(map(int,hlt_list))
	else:
		hidden_topology=[]
	hlt_list=filter(None,decoder_topology.split(','))
	if hlt_list:
		decoder_topology=list(map(int,hlt_list))
	else:
		decoder_topology=[]

	hidden_caps_layers=[]
	include_last=False

	ma=MethylationArray.from_pickle(train_methyl_array)
	ma_v=MethylationArray.from_pickle(val_methyl_array)
	if test_methyl_array and predict:
		ma_t=MethylationArray.from_pickle(test_methyl_array)

	try:
		ma.remove_na_samples(interest_col)
		ma_v.remove_na_samples(interest_col)
		if test_methyl_array and predict:
			ma_t.remove_na_samples(interest_col)
	except:
		pass

	if select_subtypes:
		print(ma.pheno[interest_col].unique())
		ma.pheno=ma.pheno.loc[ma.pheno[interest_col].isin(select_subtypes)]
		ma.beta=ma.beta.loc[ma.pheno.index]
		ma_v.pheno=ma_v.pheno.loc[ma_v.pheno[interest_col].isin(select_subtypes)]
		ma_v.beta=ma_v.beta.loc[ma_v.pheno.index]
		print(ma.pheno[interest_col].unique())

		if test_methyl_array and predict:
			ma_t.pheno=ma_t.pheno.loc[ma_t.pheno[interest_col].isin(select_subtypes)]
			ma_t.beta=ma_t.beta.loc[ma_t.pheno.index]

	if custom_capsule_file2 and os.path.exists(custom_capsule_file2):
		capsules_dict=torch.load(custom_capsule_file2)
		final_modules, modulecpgs, module_names=capsules_dict['final_modules'], capsules_dict['modulecpgs'], capsules_dict['module_names']
		if min_capsule_len>1:
			include_capsules=[len(x)>min_capsule_len for x in final_modules]
			final_modules=[final_modules[i] for i in range(len(final_modules)) if include_capsules[i]]
			module_names=[module_names[i] for i in range(len(module_names)) if include_capsules[i]]
			modulecpgs=(reduce(np.union1d,final_modules)).tolist()

	else:
		final_modules, modulecpgs, module_names=build_capsules(capsule_choice,
																overlap,
																bin_len,
																ma,
																include_last,
																min_capsule_len,
																custom_capsule_file,
																gsea_superset,
																tissue,
																gene_context,
																use_set,
																number_sets,
																limited_capsule_names_file)
		if custom_capsule_file2:
			torch.save(dict(final_modules=final_modules, modulecpgs=modulecpgs, module_names=module_names),custom_capsule_file2)

	assert len(final_modules) >= min_capsules , "Below the number of allowed capsules."

	if fit_pas:
		modulecpgs=list(reduce(lambda x,y:x+y,final_modules))

	print(ma.beta.isnull().sum().sum())
	if not include_last: # ERROR HAPPENS HERE!
		ma.beta=ma.beta.loc[:,modulecpgs]
		ma_v.beta=ma_v.beta.loc[:,modulecpgs]
		if test_methyl_array and predict:
			ma_t.beta=ma_t.beta.loc[:,modulecpgs]
	print(ma.beta.isnull().sum().sum())
	# https://github.com/higgsfield/Capsule-Network-Tutorial/blob/master/Capsule%20Network.ipynb
	original_interest_col=interest_col
	if n_bins:
		new_interest_col=interest_col+'_binned'
		ma.pheno.loc[:,new_interest_col],bins=pd.cut(ma.pheno[interest_col],bins=n_bins,retbins=True)
		ma_v.pheno.loc[:,new_interest_col],_=pd.cut(ma_v.pheno[interest_col],bins=bins,retbins=True)
		if test_methyl_array and predict:
			ma_t.pheno.loc[:,new_interest_col],_=pd.cut(ma_t.pheno[interest_col],bins=bins,retbins=True)
		interest_col=new_interest_col

	datasets=dict()

	datasets['train']=MethylationDataset(ma,interest_col,modules=final_modules, module_names=module_names, original_interest_col=original_interest_col, run_pas=fit_pas)
	print(datasets['train'].X.isnull().sum().sum())
	datasets['val']=MethylationDataset(ma_v,interest_col,modules=final_modules, module_names=module_names, original_interest_col=original_interest_col, run_pas=fit_pas)
	if test_methyl_array and predict:
		datasets['test']=MethylationDataset(ma_t,interest_col,modules=final_modules, module_names=module_names, original_interest_col=original_interest_col, run_pas=fit_pas)

	dataloaders=dict()

	dataloaders['train']=DataLoader(datasets['train'],batch_size=batch_size,shuffle=True,num_workers=8, pin_memory=True, drop_last=True)
	dataloaders['val']=DataLoader(datasets['val'],batch_size=batch_size,shuffle=False,num_workers=8, pin_memory=True, drop_last=False)
	n_primary=len(final_modules)
	if test_methyl_array and predict:
		dataloaders['test']=DataLoader(datasets['test'],batch_size=batch_size,shuffle=False,num_workers=8, pin_memory=True, drop_last=False)

	n_inputs=list(map(len,final_modules))

	n_out_caps=len(datasets['train'].y_unique)

	if not fit_pas:
		print("Not fitting MethylSPWNet")
		primary_caps = PrimaryCaps(modules=final_modules,hidden_topology=hidden_topology,n_output=primary_caps_out_len)
		hidden_caps = []
		output_caps = CapsLayer(n_out_caps,n_primary,primary_caps_out_len,caps_out_len,routing_iterations=routing_iterations)
		decoder = Decoder(n_out_caps*caps_out_len,len(list(ma.beta)),decoder_topology)
		model = CapsNet(primary_caps, hidden_caps, output_caps, decoder, gamma=gamma)

		if test_methyl_array and predict:
			model.load_state_dict(torch.load('capsnet_model.pkl'))


	else:
		print("Fitting MethylSPWNet")
		module_lens=[len(x) for x in final_modules]
		model=MethylSPWNet(module_lens, hidden_topology, dropout_p=0.2, n_output=n_out_caps)
		if test_methyl_array and predict:
			model.load_state_dict(torch.load('pasnet_model.pkl'))

	if torch.cuda.is_available():
		model=model.cuda()


	# extract all c_ij for all layers across all batches, or just last batch

	if l1_l2 and fit_pas:
		l1,l2=list(map(float,l1_l2.split(',')))
	elif fit_pas:
		l1,l2=0.,0.

	trainer=Trainer(model=model,
					validation_dataloader=dataloaders['val'],
					n_epochs=n_epochs,
					lr=learning_rate,
					n_primary=n_primary,
					custom_loss=custom_loss,
					gamma2=gamma2,
					pas_mode=fit_pas,
					l1=l1 if fit_pas else 0.,
					l2=l2 if fit_pas else 0.)

	if not predict:
		try:
			#assert 1==2
			trainer.fit(dataloader=dataloaders['train'])
			val_loss=min(trainer.val_losses)
			torch.save(trainer.model.state_dict(),'capsnet_model.pkl' if not fit_pas else 'pasnet_model.pkl')
			if fit_pas:
				torch.save(dict(final_modules=final_modules, modulecpgs=modulecpgs, module_names=module_names), 'pasnet_capsules.pkl')
				torch.save(dict(module_names=module_names,module_lens=module_lens,dropout_p=0.2,hidden_topology=hidden_topology,n_output=n_out_caps),'pasnet_config.pkl')
		except Exception as e:
			print(e)
			val_loss=-2

		with sqlite3.connect('jobs.db', check_same_thread=False) as conn:
			pd.DataFrame([job,val_loss],index=['job','val_loss'],columns=[0]).T.to_sql('val_loss',conn,if_exists='append')
	else:
		if test_methyl_array:
			trainer.weights=1.
			Y=trainer.predict(dataloaders['test'])
			pickle.dump(Y,open('predictions.pkl','wb'))
			val_loss=-1
	#print(val_loss)
	# print([min(trainer.val_losses),n_epochs,
	# 		n_bins,
	# 		bin_len,
	# 		min_capsule_len,
	# 		primary_caps_out_len,
	# 		caps_out_len,
	# 		hidden_topology,
	# 		gamma,
	# 		decoder_topology,
	# 		learning_rate,
	# 		routing_iterations])



	return val_loss
