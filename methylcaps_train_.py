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
from methylcaps_data_models import *
import sqlite3
RANDOM_SEED=42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#@pysnooper.snoop('get_mod.log')
def get_final_modules(ma=None,a='450kannotations.bed',b='lola_vignette_data/activeDHS_universe.bed', include_last=False, min_capsule_len=2000):
	allcpgs=ma.beta.columns.values
	df=BedTool(a).to_dataframe()
	df.iloc[:,0]=df.iloc[:,0].astype(str).map(lambda x: 'chr'+x.split('.')[0])
	df=df.set_index('name').loc[list(ma.beta)].reset_index().iloc[:,[1,2,3,0]]
	df_bed=pd.read_table(b,header=None)
	df_bed['features']=np.arange(df_bed.shape[0])
	df_bed=df_bed.iloc[:,[0,1,2,-1]]
	b=BedTool.from_dataframe(df)
	a=BedTool.from_dataframe(df_bed)#('lola_vignette_data/activeDHS_universe.bed')
	c=a.intersect(b,wa=True,wb=True).sort()
	d=c.groupby(g=[1,2,3,4],c=(8,8),o=('count','distinct'))
	df2=d.to_dataframe()
	df3=df2.loc[df2.iloc[:,-2]>min_capsule_len]
	modules = [cpgs.split(',') for cpgs in df3.iloc[:,-1].values]
	modulecpgs=np.array(list(set(list(reduce(lambda x,y:x+y,modules)))))
	if include_last:
		missing_cpgs=np.setdiff1d(allcpgs,modulecpgs).tolist()
	final_modules = modules+([missing_cpgs] if include_last else [])
	module_names=(df3.iloc[:,0]+'_'+df3.iloc[:,1].astype(str)+'_'+df3.iloc[:,2].astype(str)).tolist()
	return final_modules,modulecpgs,module_names

def train_capsnet_(train_methyl_array,
					val_methyl_array,
					interest_col,
					n_epochs,
					n_bins,
					bin_len,
					min_capsule_len,
					primary_caps_out_len,
					caps_out_len,
					hidden_topology,
					gamma,
					decoder_topology,
					learning_rate,
					routing_iterations,
					overlap,
					custom_loss,
					gamma2,
					job):

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

	overlap=int(overlap*bin_len)

	if not os.path.exists('hg19.{}.overlap.{}.bed'.format(bin_len,overlap)):
		BedTool('hg19.genome').makewindows(g='hg19.genome',w=bin_len,s=bin_len-overlap).saveas('hg19.{}.overlap.{}.bed'.format(bin_len,overlap))#.to_dataframe().shape

	ma=MethylationArray.from_pickle(train_methyl_array)
	ma_v=MethylationArray.from_pickle(val_methyl_array)

	try:
		ma.remove_na_samples(interest_col)
		ma_v.remove_na_samples(interest_col)
	except:
		pass

	final_modules,modulecpgs,module_names=get_final_modules(ma=ma,b='hg19.{}.overlap.{}.bed'.format(bin_len,overlap),include_last=include_last, min_capsule_len=min_capsule_len)
	print('LEN_MODULES',len(final_modules))

	if not include_last:
		ma.beta=ma.beta.loc[:,modulecpgs]
		ma_v.beta=ma_v.beta.loc[:,modulecpgs]
	# https://github.com/higgsfield/Capsule-Network-Tutorial/blob/master/Capsule%20Network.ipynb
	original_interest_col=interest_col
	if n_bins:
		new_interest_col=interest_col+'_binned'
		ma.pheno.loc[:,new_interest_col],bins=pd.cut(ma.pheno[interest_col],bins=n_bins,retbins=True)
		ma_v.pheno.loc[:,new_interest_col],bins=pd.cut(ma_v.pheno[interest_col],bins=bins,retbins=True)
		original_interest_col,interest_col=original_interest_col,new_interest_col

	dataset=MethylationDataset(ma,interest_col,modules=final_modules, module_names=module_names, original_interest_col=original_interest_col)
	dataset_v=MethylationDataset(ma_v,interest_col,modules=final_modules, module_names=module_names, original_interest_col=original_interest_col)

	dataloader=DataLoader(dataset,batch_size=16,shuffle=True,num_workers=8, drop_last=True)
	dataloader_v=DataLoader(dataset_v,batch_size=16,shuffle=False,num_workers=8, drop_last=False)

	n_inputs=list(map(len,final_modules))
	n_primary=len(final_modules)

	primary_caps = PrimaryCaps(modules=final_modules,hidden_topology=hidden_topology,n_output=primary_caps_out_len)
	hidden_caps = []
	n_out_caps=len(dataset.y_unique)

	output_caps = CapsLayer(n_out_caps,n_primary,primary_caps_out_len,caps_out_len,routing_iterations=routing_iterations)
	decoder = Decoder(n_out_caps*caps_out_len,len(list(ma.beta)),decoder_topology)
	capsnet = CapsNet(primary_caps, hidden_caps, output_caps, decoder, gamma=gamma)

	if torch.cuda.is_available():
		capsnet=capsnet.cuda()


	# extract all c_ij for all layers across all batches, or just last batch

	trainer=Trainer(capsnet=capsnet,
					validation_dataloader=dataloader_v,
					n_epochs=n_epochs,
					lr=learning_rate,
					n_primary=n_primary,
					custom_loss=custom_loss,
					gamma2=gamma2)

	try:
		#assert 1==2
		trainer.fit(dataloader=dataloader)
		val_loss=min(trainer.val_losses)
	except Exception as e:
		print(e)
		val_loss=-2
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

	with sqlite3.connect('jobs.db', check_same_thread=False) as conn:
		pd.DataFrame([job,val_loss],index=['job','val_loss'],columns=[0]).T.to_sql('val_loss',conn,if_exists='append')

	return val_loss
