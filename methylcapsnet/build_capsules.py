import pandas as pd
from pymethylprocess.MethylationDataTypes import MethylationArray
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")
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
# from methylcapsnet.methylcaps_data_models import *
import sqlite3
import os
import glob
import dask
import methyl_capsules
from dask.diagnostics import ProgressBar
from pathos.multiprocessing import Pool
import multiprocessing
import dask.bag as db
from pybedtools import BedTool
from distributed import Client, LocalCluster, get_task_stream
RANDOM_SEED=42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def print_if_exists(f):
	if os.path.exists(f):
		print('{} does exist'.format(f))
	else:
		print('{} does not exist'.format(f))

methylcaps_dir=os.path.dirname(methyl_capsules.__file__)
annotations450 = os.path.abspath(os.path.join(methylcaps_dir, 'data/450kannotations.bed'))
hg19 = os.path.abspath(os.path.join(methylcaps_dir, 'data/hg19.genome'))
selected_caps_file = os.path.abspath(os.path.join(methylcaps_dir, 'data/selected_capsules.p'))

gsea_collections = os.path.abspath(os.path.join(methylcaps_dir, 'data/gsea_collections.symbols.p'))
gene_set_weights = {os.path.basename(f).split('_')[1]: f for f in glob.glob(os.path.abspath(os.path.join(methylcaps_dir, 'data/SetTestWeights_*.txt')))}
gene2cpg = os.path.abspath(os.path.join(methylcaps_dir, 'data/gene2cpg.p'))

CAPSULES=['gene',
			'gene_context',
			'GSEA_C5.BP',
			'GSEA_C6',
			'GSEA_C1',
			'GSEA_H',
			'GSEA_C3.MIR',
			'GSEA_C2.CGP',
			'GSEA_C4.CM',
			'GSEA_C5.CC',
			'GSEA_C3.TFT',
			'GSEA_C5.MF',
			'GSEA_C7',
			'GSEA_C2.CP',
			'GSEA_C4.CGN',
			'UCSC_RefGene_Name',
			'UCSC_RefGene_Accession',
			'UCSC_RefGene_Group',
			'UCSC_CpG_Islands_Name',
			'Relation_to_UCSC_CpG_Island',
			'Phantom',
			'DMR',
			'Enhancer',
			'HMM_Island',
			'Regulatory_Feature_Name',
			'Regulatory_Feature_Group',
			'DHS']

final_caps_files = {k: os.path.abspath(os.path.join(methylcaps_dir, 'data/final_capsules__{}.p'.format(k))) for k in CAPSULES }


if 0:
	print_if_exists(annotations450)
	print_if_exists(hg19)
	print_if_exists(selected_caps_file)

# @pysnooper.snoop('get_mod.log')
def get_binned_modules(ma=None,a=annotations450,b='lola_vignette_data/activeDHS_universe.bed', include_last=False, min_capsule_len=2000):
	allcpgs=ma.beta.columns.values
	a=BedTool(a)
	b=BedTool(b)
	# a.saveas('a.bed')
	# b.saveas('b.bed')
	a_orig=a
	df=BedTool(a).to_dataframe()
	df.iloc[:,0]=df.iloc[:,0].astype(str)#.map(lambda x: 'chr'+x.split('.')[0])
	df=df.set_index('name').loc[list(ma.beta)].reset_index().iloc[:,[1,2,3,0]]
	a=BedTool.from_dataframe(df)
	# df_bed=pd.read_table(b,header=None)
	# df_bed['features']=np.arange(df_bed.shape[0])
	# df_bed=df_bed.iloc[:,[0,1,2,-1]]
	# b=BedTool.from_dataframe(df)
	# a=BedTool.from_dataframe(df_bed)#('lola_vignette_data/activeDHS_universe.bed')
	df_bed=BedTool(b).to_dataframe()
	if df_bed.shape[1]<4:
		df_bed['features']=np.arange(df_bed.shape[0])
	b=BedTool.from_dataframe(df_bed)
	try:
		c=b.intersect(a,wa=True,wb=True).sort()
		# c.saveas('c.bed')
		d=c.groupby(g=[1,2,3,4],c=(8,8),o=('count','distinct'))
	except:
		df=BedTool(a_orig).to_dataframe()
		df.iloc[:,0]=df.iloc[:,0].astype(str).map(lambda x: 'chr'+x.split('.')[0])
		df=df.set_index('name').loc[list(ma.beta)].reset_index().iloc[:,[1,2,3,0]]
		a=BedTool.from_dataframe(df)
		c=b.intersect(a,wa=True,wb=True).sort()
		# c.saveas('c.bed')
		d=c.groupby(g=[1,2,3,4],c=(8,8),o=('count','distinct'))
	#d.saveas('d.bed')
	df2=d.to_dataframe()
	df3=df2.loc[df2.iloc[:,-2]>min_capsule_len]
	modules = [cpgs.split(',') for cpgs in df3.iloc[:,-1].values]
	modulecpgs=np.array(list(set(list(reduce(lambda x,y:x+y,modules)))))
	if include_last:
		missing_cpgs=np.setdiff1d(allcpgs,modulecpgs).tolist()
	final_modules = modules+([missing_cpgs] if include_last else [])
	module_names=(df3.iloc[:,0]+'_'+df3.iloc[:,1].astype(str)+'_'+df3.iloc[:,2].astype(str)).tolist()
	return final_modules,modulecpgs.tolist(),module_names

def return_caps(capsule,allcpgs,min_capsule_len):
	capsule=np.intersect1d(capsule,allcpgs).tolist()
	if len(capsule)>=min_capsule_len:
		return capsule
	else:
		return []

#@pysnooper.snoop('reduce_caps.log')
def reduce_caps(capsules,allcpgs,min_capsule_len):
	cluster = LocalCluster(n_workers=multiprocessing.cpu_count()*2, threads_per_worker=20)
	client = Client(cluster)
	capsule_names=list(capsules.keys())

	capsules_bag=db.from_sequence(list(capsules.values()))
	capsules_intersect=capsules_bag.map(lambda x: np.intersect1d(x,allcpgs))
	capsules_len=capsules_intersect.map(lambda x: x if len(x) >= min_capsule_len else [])
	# with get_task_stream(plot='save', filename="task-stream.html") as ts:
	capsules=capsules_len.compute()
	capsules=dict([(capsule_names[i],capsules[i].tolist()) for i in range(len(capsule_names)) if len(capsules[i])])
	#print(list(capsules.keys()))
	client.close()
	return capsules

@pysnooper.snoop('get_caps.log')
def return_custom_capsules(ma=None,capsule_file=selected_caps_file, capsule_sets=['all'], min_capsule_len=2000, include_last=False, limited_capsule_names_file=''):
	allcpgs=ma.beta.columns.values
	if limited_capsule_names_file:
		with open(limited_capsule_names_file) as f:
			limited_capsule_names=f.read().replace('\n',' ').split()
	else:
		limited_capsule_names=[]
	caps_dict=pickle.load(open(capsule_file,'rb'))
	capsules={}
	if 'all' in capsule_sets:
		capsule_sets=list(caps_dict.keys())

	for caps_set in capsule_sets:
		if limited_capsule_names_file:
			capsule_list=np.intersect1d(list(caps_dict[caps_set].keys()),limited_capsule_names).tolist()
		else:
			capsule_list=list(caps_dict[caps_set].keys())
		for capsule in capsule_list:
			capsules[capsule]=caps_dict[caps_set][capsule]#dask.delayed(lambda x:return_caps(x,allcpgs,min_capsule_len))(caps_dict[caps_set][capsule])


	capsules=reduce_caps(capsules,allcpgs,min_capsule_len)
	#capsules=dask.compute(capsules,scheduler='threading')[0]
	#capsules={capsule:capsules[capsule] for capsule in capsules if capsules[capsule]}
	modules = [capsules[capsule] for capsule in capsules if capsules[capsule]]
	modulecpgs=reduce(np.union1d,modules)#np.array(list(set(list(reduce(lambda x,y:x+y,modules)))))
	module_names=list(capsules.keys())#(df3.iloc[:,0]+'_'+df3.iloc[:,1].astype(str)+'_'+df3.iloc[:,2].astype(str)).tolist()
	return modules,modulecpgs,module_names

def divide_chunks(l, n):
	for i in range(0, len(l), len(l)//n):
		yield l[i:i + n]

@pysnooper.snoop('gsea_build.log')
def return_gsea_capsules(ma=None,tissue='',context_on=False,use_set=False,gsea_superset='H',n_top_sets=25,min_capsule_len=2000, all_genes=False, union_cpgs=True, limited_capsule_names_file=''):
	global gene2cpg, gsea_collections, gene_set_weights
	if limited_capsule_names_file:
		with open(limited_capsule_names_file) as f:
			limited_capsule_names=f.read().replace('\n',' ').split()
	else:
		limited_capsule_names=[]
	allcpgs=ma.beta.columns.values
	entire_sets=use_set
	collection=gsea_superset
	gene2cpg=pickle.load(open(gene2cpg,'rb'))
	if all_genes:
		gene_sets=list(gene2cpg.keys())
	else:
		gsea=pickle.load(open(gsea_collections,'rb'))
		if tissue:
			gene_sets=pd.read_csv(gene_set_weights[collection],sep='\t',index_col=0)
			if tissue!='ubiquitous':
				gene_sets=(gene_sets.quantile(1.,axis=1)-gene_sets.quantile(0.,axis=1)).sort_values().index.tolist()
			else:
				gene_sets=gene_sets[tissue].sort_values(ascending=False).index.tolist()
		else:
			gene_sets=list(gsea[collection].keys())
	intersect_context=False
	if limited_capsule_names_file:
		gene_sets_tmp=np.intersect1d(gene_sets,limited_capsule_names).tolist()
		print('LIMITED GENE CAPS',gene_sets_tmp)
		if gene_sets_tmp:
			gene_sets=gene_sets_tmp
			intersect_context=True
	if not tissue:
		n_top_sets=0
	if n_top_sets and not all_genes:
		gene_sets=gene_sets[:n_top_sets]

	capsules=dict()
	if all_genes:
		entire_sets=False
	if entire_sets:
		context_on=False

	def process_gene_set(gene_set):
		capsules=[]
		gene_set_cpgs=[]
		for genename in (gsea[collection][gene_set] if not all_genes else [gene_set]):
			gene=gene2cpg.get(genename,{'Gene':[],'Upstream':[]})
			if context_on:
				for k in ['Gene','Upstream']:
					context=gene.get(k,[])
					if len(context):
						capsules.append(('{}_{}'.format(genename,k),list(context)))
						#capsules['{}_{}'.format(genename,k)]=context.tolist()
			else:
				if not entire_sets:
					capsules.append((genename,np.union1d(gene.get('Gene',[]),gene.get('Upstream',[])).tolist()))
					#capsules[genename]=np.union1d(gene.get('Gene',[]),gene.get('Upstream',[])).tolist()
				else:
					upstream=gene.get('Upstream',[])
					gene=gene.get('Gene',[])
					cpg_set=np.union1d(gene,upstream)
					if cpg_set.tolist():
						gene_set_cpgs.append(cpg_set)
		if entire_sets and not all_genes:
			capsules.append((gene_set,reduce(np.union1d,gene_set_cpgs).tolist()))
			#capsules[gene_set]=reduce(np.union1d,gene_set_cpgs).tolist()
		return capsules

	def process_chunk(chunk):
		with ProgressBar():
			chunk=dask.compute(*chunk,scheduler='threading')
		return chunk

	with ProgressBar():
		capsules=dict(list(reduce(lambda x,y: x+y,dask.compute(*[dask.delayed(process_gene_set)(gene_set) for gene_set in gene_sets],scheduler='threading'))))


	capsules2=[]
	#caps_lens=np.array([len(capsules[capsule]) for capsule in capsules])

	# cluster = LocalCluster(n_workers=multiprocessing.cpu_count()*2, threads_per_worker=20)
	# client = Client(cluster)
	capsule_names=list(capsules.keys())

	if intersect_context:
		capsules_tmp_names=np.intersect1d(capsule_names,limited_capsule_names).tolist()
		if capsules_tmp_names:
			capsules={k:capsules[k] for k in capsules_tmp_names}
			capsule_names=capsules_tmp_names

	capsules=reduce_caps(capsules,allcpgs,min_capsule_len)

	# print(capsule_names)
	# capsules_bag=db.from_sequence(list(capsules.values()))
	# capsules_intersect=capsules_bag.map(lambda x: np.intersect1d(x,allcpgs))
	# capsules_len=capsules_intersect.map(lambda x: x if len(x) >= min_capsule_len else [])
	# # with get_task_stream(plot='save', filename="task-stream.html") as ts:
	# capsules=capsules_len.compute()
	# #print(capsules)
	# capsules=dict([(capsule_names[i],capsules[i].tolist()) for i in range(len(capsule_names)) if len(capsules[i])])

	# for capsule in capsules:
	# 	capsules2.append([capsule,dask.delayed(return_caps)(capsules[capsule],allcpgs,min_capsule_len)])
	# cpus=multiprocessing.cpu_count()
	# caps_chunks=list(divide_chunks(capsules2,cpus))
	# p=Pool(cpus)
	# capsules=dict(list(reduce(lambda x,y: x+y,p.map(process_chunk,caps_chunks))))

	# with ProgressBar():
	# 	capsules=dask.compute(capsules2,scheduler='threading')[0]
	#print(capsules)
	modules = list(capsules.values())#[capsules[capsule] for capsule in capsules if capsules[capsule]]
	modulecpgs=reduce((np.union1d if union_cpgs else (lambda x,y:x+y)),modules).tolist()
	module_names=list(capsules.keys())

	return modules,modulecpgs,module_names

def get_gene_sets(cpgs,final_capsules,collection,tissue,n_top_sets):
	global gsea_collections, gene_set_weights
	gsea=pickle.load(open(gsea_collections,'rb'))
	if tissue:
		gene_sets=pd.read_csv(gene_set_weights[collection],sep='\t',index_col=0)
		if tissue!='ubiquitous':
			gene_sets=(gene_sets.quantile(1.,axis=1)-gene_sets.quantile(0.,axis=1)).sort_values().index.tolist()
		else:
			gene_sets=gene_sets[tissue].sort_values(ascending=False).index.tolist()
	else:
		gene_sets=list(gsea[collection].keys())
	if n_top_sets:
		gene_sets=gene_sets[:n_top_sets]
	final_capsules=final_capsules['GSEA_{}'.format(collection)]
	final_capsules=final_capsules[final_capsules['cpg'].isin(cpgs)]
	return final_capsules[final_capsules['feature'].isin(gene_sets)]['cpg'].values

#@pysnooper.snoop('final_caps.log')
def return_final_capsules(methyl_array, capsule_choice, min_capsule_len, collection,tissue, n_top_sets, limited_capsule_names_file, gsea_superset, return_original_capsule_assignments=False):
	from sklearn.preprocessing import LabelEncoder
	global final_caps_files
	if limited_capsule_names_file:
		with open(limited_capsule_names_file) as f:
			limited_capsule_names=f.read().replace('\n',' ').split()
	else:
		limited_capsule_names=[]
	#final_capsules=pickle.load(open(final_caps_files[capsule_choice],'rb'))
	if len(capsule_choice)>1:
		cpg_arr=pd.concat([pd.read_pickle(final_caps_files[caps_choice]) for caps_choice in capsule_choice])
	else:
		cpg_arr=pd.read_pickle(final_caps_files[capsule_choice[0]])
	cpgs=np.intersect1d(methyl_array.beta.columns.values,cpg_arr['cpg'].values)
	if gsea_superset:
		cpgs=get_gene_sets(cpgs,cpg_arr,gsea_superset,tissue,n_top_sets)
	if limited_capsule_names:
		print(limited_capsule_names)
		cpg_arr=cpg_arr[cpg_arr['feature'].isin(limited_capsule_names)]#cpgs=np.intersect1d(cpgs,cpg_arr[cpg_arr['feature'].isin(limited_capsule_names)]['cpg'].values)
	cpg_arr=cpg_arr[cpg_arr['cpg'].isin(cpgs)]
	capsules=[]
	cpgs=[]
	features=[]
	cpg_arr=pd.DataFrame(cpg_arr.groupby('feature').filter(lambda x: len(x['cpg'])>=min_capsule_len))
	# for name, dff in .groupby('feature'):
	# 	cpg=dff['cpg'].values
	# 	capsules.append(cpg)
	# 	cpgs.extend(cpg.tolist())
	# 	features.append(name)
	cpgs,features=cpg_arr['cpg'].values,cpg_arr['feature'].unique()
	split_idx=np.cumsum(np.bincount(LabelEncoder().fit_transform(cpg_arr['feature'].values).flatten().astype(int)).flatten().astype(int)).flatten().astype(int)[:-1]
	capsules=np.split(cpgs,split_idx)
	# print(capsules)
	if return_original_capsule_assignments:
		return capsules,cpgs,features,cpg_arr[['feature','cpg']]
	else:
		return capsules,cpgs,features#cpg_arr['feature'].unique()#cpg_arr['cpg'].values

@pysnooper.snoop('build_caps.log')
def build_capsules(capsule_choice,
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
					limited_capsule_names_file):
	capsules,finalcpgs,capsule_names=[],[],[]
	annotation_file=annotations450
	if 'genomic_binned' in capsule_choice:
		overlap=int(overlap*bin_len)
		genome_file=hg19
		gname=os.path.basename(genome_file).split('.')[0]
		overlap_file='{}.{}.overlap.{}.bed'.format(gname,bin_len,overlap)
		if not os.path.exists(overlap_file):
			BedTool(genome_file).makewindows(g=genome_file,w=bin_len,s=bin_len-overlap).saveas('{}.{}.overlap.{}.bed'.format(gname,bin_len,overlap))#.to_dataframe().shape
		print(annotation_file,overlap_file)
		final_modules,modulecpgs,module_names=get_binned_modules(ma=ma,a=annotation_file,b=overlap_file,include_last=include_last, min_capsule_len=min_capsule_len)
		print('LEN_MODULES',len(final_modules))
		capsules.extend(final_modules)
		finalcpgs.extend(modulecpgs)
		capsule_names.extend(module_names)

	if 'custom_bed' in capsule_choice:
		final_modules,modulecpgs,module_names=get_binned_modules(ma=ma,a=annotation_file,b=custom_capsule_file,include_last=include_last, min_capsule_len=min_capsule_len)
		capsules.extend(final_modules)
		finalcpgs.extend(modulecpgs)
		capsule_names.extend(module_names)

	if 'custom_set' in capsule_choice:
		final_modules,modulecpgs,module_names=return_custom_capsules(ma=ma,capsule_file=custom_capsule_file, capsule_sets=['all'], min_capsule_len=min_capsule_len, include_last=include_last)
		capsules.extend(final_modules)
		finalcpgs.extend(modulecpgs)
		capsule_names.extend(module_names)

	if np.intersect1d(CAPSULES,capsule_choice).tolist():
		final_modules,modulecpgs,module_names=return_final_capsules(methyl_array=ma, capsule_choice=capsule_choice, min_capsule_len=min_capsule_len, collection=gsea_superset,tissue=tissue, n_top_sets=number_sets, limited_capsule_names_file=limited_capsule_names_file, gsea_superset=gsea_superset)
		capsules.extend(final_modules)
		finalcpgs.extend(modulecpgs)
		capsule_names.extend(module_names)

	# if 0:
	#
	# 	selected_sets=np.intersect1d(['UCSC_RefGene_Name','UCSC_RefGene_Accession', 'UCSC_RefGene_Group', 'UCSC_CpG_Islands_Name', 'Relation_to_UCSC_CpG_Island', 'Phantom', 'DMR', 'Enhancer', 'HMM_Island', 'Regulatory_Feature_Name', 'Regulatory_Feature_Group', 'DHS'],capsule_choice).tolist()
	# 	if selected_sets:
	# 		final_modules,modulecpgs,module_names=return_custom_capsules(ma=ma,capsule_file=selected_caps_file, capsule_sets=selected_sets, min_capsule_len=min_capsule_len, include_last=include_last, limited_capsule_names_file=limited_capsule_names_file)
	# 		capsules.extend(final_modules)
	# 		finalcpgs.extend(modulecpgs)
	# 		capsule_names.extend(module_names)
	#
	# 	gsea_bool=(("GSEA" in capsule_choice and gsea_superset) or 'all_gene_sets' in capsule_choice)
	#
	# 	if gsea_bool:
	# 		final_modules,modulecpgs,module_names=return_gsea_capsules(ma=ma,tissue=tissue,context_on=gene_context,use_set=use_set,gsea_superset=gsea_superset,n_top_sets=number_sets,min_capsule_len=min_capsule_len, all_genes=('all_gene_sets' in capsule_choice), limited_capsule_names_file=limited_capsule_names_file)
	# 		capsules.extend(final_modules)
	# 		finalcpgs.extend(modulecpgs)
	# 		capsule_names.extend(module_names)

	final_modules=capsules
	modulecpgs=list(set(finalcpgs))
	module_names=capsule_names

	# if limited_capsule_names_file and not (selected_sets or gsea_bool):
	# 	with open(limited_capsule_names_file) as f:
	# 		limited_capsule_names=f.read().replace('\n',' ').split()
	# 	capsules=[]
	# 	capsule_names=[]
	# 	for i in range(len(module_names)):
	# 		if module_names[i] in limited_capsule_names:
	# 			capsule_names.append(module_names[i])
	# 			capsules.append(final_modules[i])
	#
	# 	modulecpgs=list(set(list(reduce(lambda x,y: x+y,capsules))))
	# 	final_modules=capsules
	# 	module_names=capsule_names

	print("{} modules, {} cpgs, {} module names, {} missing".format(len(final_modules),len(modulecpgs),len(module_names),ma.beta.isnull().sum().sum()))

	return final_modules, modulecpgs, module_names
