from methylcapsnet.samplers import ImbalancedDatasetSampler
from pymethylprocess.MethylationDataTypes import MethylationArray
import numpy as np, pandas as pd
from captum.attr import GradientShap
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler
from methylcapsnet.methylcaps_data_models import *

def return_spw_importances_(train_methyl_array,
							val_methyl_array,
							interest_col,
							select_subtypes,
							capsules_pickle,
							include_last,
							n_bins,
							spw_config,
							model_state_dict_pkl,
							batch_size,
							by_subtype=False
							):
	ma=MethylationArray.from_pickle(train_methyl_array)
	ma_v=MethylationArray.from_pickle(val_methyl_array)

	try:
		ma.remove_na_samples(interest_col)
		ma_v.remove_na_samples(interest_col)
	except:
		pass

	if select_subtypes:
		ma.pheno=ma.pheno.loc[ma.pheno[interest_col].isin(select_subtypes)]
		ma.beta=ma.beta.loc[ma.pheno.index]
		ma_v.pheno=ma_v.pheno.loc[ma_v.pheno[interest_col].isin(select_subtypes)]
		ma_v.beta=ma_v.beta.loc[ma_v.pheno.index]

	capsules_dict=torch.load(capsules_pickle)

	final_modules, modulecpgs, module_names = capsules_dict['final_modules'],capsules_dict['modulecpgs'],capsules_dict['module_names']

	if not include_last:
		ma.beta=ma.beta.loc[:,modulecpgs]
		ma_v.beta=ma_v.beta.loc[:,modulecpgs]

	original_interest_col=interest_col

	if n_bins:
		new_interest_col=interest_col+'_binned'
		ma.pheno.loc[:,new_interest_col],bins=pd.cut(ma.pheno[interest_col],bins=n_bins,retbins=True)
		ma_v.pheno.loc[:,new_interest_col],_=pd.cut(ma_v.pheno[interest_col],bins=bins,retbins=True)
		interest_col=new_interest_col

	datasets=dict()
	datasets['train']=MethylationDataset(ma,interest_col,modules=final_modules, module_names=module_names, original_interest_col=original_interest_col, run_spw=True)
	datasets['val']=MethylationDataset(ma_v,interest_col,modules=final_modules, module_names=module_names, original_interest_col=original_interest_col, run_spw=True)

	y_val=datasets['val'].y_label
	y_val_uniq=np.unique(y_val)

	dataloaders=dict()
	dataloaders['train']=DataLoader(datasets['train'],batch_size=batch_size,shuffle=True,num_workers=8, pin_memory=True, drop_last=True)
	dataloaders['val']=DataLoader(datasets['val'],batch_size=batch_size,shuffle=False,num_workers=8, pin_memory=True, drop_last=False)
	n_primary=len(final_modules)

	spw_config=torch.load(spw_config)
	spw_config.pop('module_names')

	model=MethylSPWNet(**spw_config)
	model.load_state_dict(torch.load(model_state_dict_pkl))

	if torch.cuda.is_available():
		model=model.cuda()

	model.eval()

	pathway_extractor=model.pathways

	#extract_pathways = lambda modules_x:torch.cat([pathway_extractor[i](module_x) for i,module_x in enumerate(modules_x)],dim=1)

	tensor_data=dict(train=dict(X=[],y=[]),val=dict(X=[],y=[]))

	for k in tensor_data:
		for i,(batch) in enumerate(dataloaders[k]):
			x = batch[0]
			y_true=batch[-1].argmax(1)#[-2]
			modules_x = batch[1:-1]#2]
			if torch.cuda.is_available():
				x=x.cuda()
				modules_x=modules_x[0].cuda()#[module.cuda() for module in modules_x]
			tensor_data[k]['X'].append(pathway_extractor(x,modules_x).detach().cpu())#extract_pathways(modules_x).detach().cpu())
			tensor_data[k]['y'].append(y_true.flatten().view(-1,1))
		tensor_data[k]['X']=torch.cat(tensor_data[k]['X'],dim=0)
		tensor_data[k]['y']=torch.cat(tensor_data[k]['y'],dim=0)
		print(tensor_data[k]['X'].size(),tensor_data[k]['y'].size())
		tensor_data[k]=TensorDataset(tensor_data[k]['X'],tensor_data[k]['y'])
		dataloaders[k]=DataLoader(tensor_data[k],batch_size=32,sampler=ImbalancedDatasetSampler(tensor_data[k]))

	model=model.output_net
	to_cuda=lambda x: x.cuda() if torch.cuda.is_available() else x
	y=np.unique(tensor_data['train'].tensors[1].numpy().flatten())
	gs = GradientShap(model)
	X_train=torch.cat([next(iter(dataloaders['train']))[0] for i in range(2)],dim=0)
	if torch.cuda.is_available():
		X_train=X_train.cuda()

	#val_loader=iter(dataloaders['val'])


	def return_importances(dataloaders, X_train):
		attributions=[]
		for i in range(20):
			batch=next(iter(dataloaders['val']))
			X_test=to_cuda(batch[0])
			y_test=to_cuda(batch[1].flatten())
			attributions.append(torch.abs(gs.attribute(X_test, stdevs=0.03, n_samples=200, baselines=X_train,
										   target=y_test, return_convergence_delta=False)))#torch.tensor(y_i)
		attributions=torch.sum(torch.cat(attributions,dim=0),dim=0)
		importances=pd.DataFrame(pd.Series(attributions.detach().cpu().numpy(),index=module_names).sort_values(ascending=False),columns=['importances'])
		return importances


	if by_subtype:
		importances=[]
		for k in y_val_uniq:
			idx=np.where(y_val==k)
			sampler=SubsetRandomSampler(idx)
			dataloaders['val']=DataLoader(tensor_data['val'],batch_size=32,sampler=sampler)
			df=return_importances(dataloaders, X_train)
			df['subtype']=k
			importances.append(df)
		importances=pd.concat(importances)
	else:
		importances=return_importances(dataloaders, X_train)

	return importances
