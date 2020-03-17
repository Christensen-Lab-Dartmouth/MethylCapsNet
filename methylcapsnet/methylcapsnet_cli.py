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
from methylcapsnet.methylcaps_data_models import *
import sqlite3
from methylcapsnet.methylcaps_model_ import *
RANDOM_SEED=42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

CONTEXT_SETTINGS = dict(help_option_names=['-h','--help'], max_content_width=90)

@click.group(context_settings= CONTEXT_SETTINGS)
@click.version_option(version='0.1')
def methylcaps():
	pass

#@pysnooper.snoop('main_snoop.log')
@methylcaps.command()
@click.option('-i', '--train_methyl_array', default='./train_val_test_sets/train_methyl_array.pkl', help='Input database for beta and phenotype data.', type=click.Path(exists=False), show_default=True)
@click.option('-v', '--val_methyl_array', default='./train_val_test_sets/val_methyl_array.pkl', help='Test database for beta and phenotype data.', type=click.Path(exists=False), show_default=True)
@click.option('-ic', '--interest_col', default='disease', help='Specify column looking to make predictions on.', show_default=True)
@click.option('-e', '--n_epochs', default=500, help='Number of epochs to train over.', show_default=True)
@click.option('-nb', '--n_bins', default=0, help='Number of bins if column is continuous variable.', show_default=True)
@click.option('-bl', '--bin_len', default=1000000, help='Length in bp of genomic regions, will add separate ability to import custom enrichment.', show_default=True)
@click.option('-mcl', '--min_capsule_len', default=350, help='Minimum number CpGs to include in capsules.', show_default=True)
@click.option('-po', '--primary_caps_out_len', default=40, help='Dimensionality of primary capsule embeddings.', show_default=True)
@click.option('-co', '--caps_out_len', default=20, help='Dimensionality of output capsule embeddings.', show_default=True)
@click.option('-ht', '--hidden_topology', default='', help='Topology of hidden layers, comma delimited, leave empty for one layer encoder, eg. 100,100 is example of 5-hidden layer topology. This topology is used for each primary capsule. Try 30,80,50?', type=click.Path(exists=False), show_default=True)
@click.option('-g', '--gamma', default=1e-2, help='How much to weight autoencoder loss.', show_default=True)
@click.option('-dt', '--decoder_topology', default='', help='Topology of decoder layers, comma delimited, leave empty for one layer decoder, eg. 100,100 is example of 5-hidden layer topology. This topology is used for the decoder. Try 100,300?', type=click.Path(exists=False), show_default=True)
@click.option('-lr', '--learning_rate', default=1e-3, help='Learning rate.', show_default=True)
@click.option('-ri', '--routing_iterations', default=3, help='Number of routing iterations.', show_default=True)
@click.option('-ov', '--overlap', default=0., help='Overlap fraction of bin length.', show_default=True)
@click.option('-cl', '--custom_loss', default='none', help='Specify custom loss function.', show_default=True, type=click.Choice(['none','cox']))
@click.option('-g2', '--gamma2', default=1e-2, help='How much to weight custom loss.', show_default=True)
@click.option('-j', '--job', default=0, help='Job number.', show_default=True)
@click.option('-cc', '--capsule_choice', default=['genomic_binned'], multiple=True, help='Specify multiple sets of capsules to include. Cannot specify both custom_bed and custom_set.', show_default=True, type=click.Choice(["GSEA",'all_gene_sets','genomic_binned','custom_bed','custom_set','gene', 'gene_context', 'GSEA_C5.BP', 'GSEA_C6', 'GSEA_C1', 'GSEA_H', 'GSEA_C3.MIR', 'GSEA_C2.CGP', 'GSEA_C4.CM', 'GSEA_C5.CC', 'GSEA_C3.TFT', 'GSEA_C5.MF', 'GSEA_C7', 'GSEA_C2.CP', 'GSEA_C4.CGN', 'UCSC_RefGene_Name', 'UCSC_RefGene_Accession', 'UCSC_RefGene_Group', 'UCSC_CpG_Islands_Name', 'Relation_to_UCSC_CpG_Island', 'Phantom', 'DMR', 'Enhancer', 'HMM_Island', 'Regulatory_Feature_Name', 'Regulatory_Feature_Group', 'DHS']))# ADD LOLA!!!
@click.option('-cf', '--custom_capsule_file', default='', help='Custom capsule file, bed or pickle.', show_default=True, type=click.Path(exists=False))
@click.option('-t', '--test_methyl_array', default='./train_val_test_sets/test_methyl_array.pkl', help='Test database for beta and phenotype data.', type=click.Path(exists=False), show_default=True)
@click.option('-pr', '--predict', is_flag=True, help='Predict on MethlyCapsNet.', show_default=True)
@click.option('-bs', '--batch_size', default=16, help='Batch size.', show_default=True)
@click.option('-lc', '--limited_capsule_names_file', default='', help='File of new line delimited names of capsules to filter from larger list.', show_default=True, type=click.Path(exists=False))
@click.option('-gsea', '--gsea_superset', default='', help='GSEA supersets.', show_default=True, type=click.Choice(['','C1', 'C3.MIR', 'C3.TFT', 'C7', 'C5.MF', 'H', 'C5.BP', 'C2.CP', 'C2.CGP', 'C4.CGN', 'C5.CC', 'C6', 'C4.CM']))
@click.option('-ts', '--tissue', default='', help='Tissue associated with GSEA.', show_default=True, type=click.Choice(['','adipose tissue','adrenal gland','appendix','bone marrow','breast','cerebral cortex','cervix, uterine','colon','duodenum','endometrium','epididymis','esophagus','fallopian tube','gallbladder','heart muscle','kidney','liver','lung','lymph node','ovary','pancreas','parathyroid gland','placenta','prostate','rectum','salivary gland','seminal vesicle','skeletal muscle','skin','small intestine','smooth muscle','spleen','stomach','testis','thyroid gland','tonsil','urinary bladder','ubiquitous']))
@click.option('-ns', '--number_sets', default=25, help='Number top gene sets to choose for tissue-specific gene sets.', show_default=True)
@click.option('-st', '--use_set', is_flag=True, help='Use sets or genes within sets.', show_default=True)
@click.option('-gc', '--gene_context', is_flag=True, help='Use upstream and gene body contexts for gsea analysis.', show_default=True)
@click.option('-ss', '--select_subtypes', default=[''], multiple=True, help='Selected subtypes if looking to reduce number of labels to predict', show_default=True)
@click.option('-fp', '--fit_spw', is_flag=True, help='Fit SPWNet for feature selection.', show_default=True)
@click.option('-l1l2', '--l1_l2', default='', help='L1, L2 penalization, comma delimited.', type=click.Path(exists=False), show_default=True)
@click.option('-cf2', '--custom_capsule_file2', default='', help='If specified as pkl, saves and loads current capsule configuration for quick use.', show_default=True, type=click.Path(exists=False))
@click.option('-mc', '--min_capsules', default=5, help='Minimum number of capsules in analysis.', show_default=True)
def model_capsnet(train_methyl_array,
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
					job,
					capsule_choice,
					custom_capsule_file,
					test_methyl_array,
					predict,
					batch_size,
					limited_capsule_names_file,
					gsea_superset,
					tissue,
					number_sets,
					use_set,
					gene_context,
					select_subtypes,
					fit_spw,
					l1_l2,
					custom_capsule_file2,
					min_capsules):

	model_capsnet_(train_methyl_array,
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
						job,
						capsule_choice,
						custom_capsule_file,
						test_methyl_array,
						predict,
						batch_size,
						limited_capsule_names_file,
						gsea_superset,
						tissue,
						number_sets,
						use_set,
						gene_context,
						list(filter(None,select_subtypes)),
						fit_spw,
						l1_l2,
						custom_capsule_file2,
						min_capsules)

@methylcaps.command()
@click.option('-pm', '--spwnet_model', default='spwnet_model.pkl', help='Path to SPWNet model to extract pathway importances.', type=click.Path(exists=False), show_default=True)
@click.option('-pcf', '--spwnet_config', default='spwnet_config.pkl', help='Path to SPWNet model configuration to extract pathway importances.', type=click.Path(exists=False), show_default=True)
@click.option('-n', '--n_capsules', default=0, help='Number pathways to extract.', show_default=True)
@click.option('-csv', '--feature_csv', default='spwnet_importances.csv', help='All extracted pathway importances.', type=click.Path(exists=False), show_default=True)
@click.option('-txt', '--capsule_txt', default='custom_capsules.txt', help='Where to extract custom capsules.', type=click.Path(exists=False), show_default=True)
def extract_capsules(spwnet_model,spwnet_config,n_capsules,feature_csv,capsule_txt):
	from methylcapsnet.methylcaps_data_models import MethylSPWNet
	spwnet_config=torch.load(spwnet_config)
	module_names=spwnet_config.pop('module_names')
	model=MethylSPWNet(**spwnet_config)
	model.load_state_dict(torch.load(spwnet_model))
	importances=model.calc_pathway_importances([len(capsule) for capsule in spwnet_config.pop('capsules')]).detach().cpu().numpy()
	importances=pd.Series(importances,index=module_names).sort_values(ascending=False)
	pd.DataFrame(importances).to_csv(feature_csv)
	with open(capsule_txt,'w') as f:
		f.write('\n'.join((importances.iloc[:n_capsules] if n_capsules else importances).index.values.tolist()))

@methylcaps.command()
@click.option('-i', '--train_methyl_array', default='./train_val_test_sets/train_methyl_array.pkl', help='Input database for beta and phenotype data.', type=click.Path(exists=False), show_default=True)
@click.option('-v', '--val_methyl_array', default='./train_val_test_sets/val_methyl_array.pkl', help='Test database for beta and phenotype data.', type=click.Path(exists=False), show_default=True)
@click.option('-ic', '--interest_col', default='disease', help='Specify column looking to make predictions on.', show_default=True)
@click.option('-ss', '--select_subtypes', default=[''], multiple=True, help='Selected subtypes if looking to reduce number of labels to predict', show_default=True)
@click.option('-cp', '--capsules_pickle', default='spwnet_capsules.pkl', help='Path to SPWNet model capsules.', type=click.Path(exists=False), show_default=True)
@click.option('-nb', '--n_bins', default=0, help='Number of bins if column is continuous variable.', show_default=True)
@click.option('-pcf', '--spw_config', default='spwnet_config.pkl', help='Path to SPWNet model configuration to extract pathway importances.', type=click.Path(exists=False), show_default=True)
@click.option('-pm', '--model_state_dict_pkl', default='spwnet_model.pkl', help='Path to SPWNet model to extract pathway importances.', type=click.Path(exists=False), show_default=True)
@click.option('-bs', '--batch_size', default=32, help='Batch size.', show_default=True)
@click.option('-n', '--n_capsules', default=0, help='Number pathways to extract.', show_default=True)
@click.option('-txt', '--capsule_txt', default='custom_capsules.txt', help='Where to extract custom capsules.', type=click.Path(exists=False), show_default=True)
@click.option('-csv', '--feature_csv', default='spwnet_importances.csv', help='All extracted pathway importances.', type=click.Path(exists=False), show_default=True)
@click.option('-st', '--by_subtype', is_flag=True, help='Interrogate each subtype.', show_default=True)
def return_spw_importances(train_methyl_array,
							val_methyl_array,
							interest_col,
							select_subtypes,
							capsules_pickle,
							n_bins,
							spw_config,
							model_state_dict_pkl,
							batch_size,
							n_capsules,
							capsule_txt,
							feature_csv,
							by_subtype):
	from methylcapsnet.methylcaps_interpret import return_spw_importances_

	importances=return_spw_importances_(train_methyl_array,
								val_methyl_array,
								interest_col,
								list(filter(None,select_subtypes)),
								capsules_pickle,
								False,
								n_bins,
								spw_config,
								model_state_dict_pkl,
								batch_size,
								by_subtype)

	importances.to_csv(feature_csv)
	with open(capsule_txt,'w') as f:
		f.write('\n'.join((importances.iloc[:n_capsules] if n_capsules else importances).index.values.tolist()))

@methylcaps.command()
@click.option('-j', '--job', default=0, help='Job number.', show_default=True)
@click.option('-l', '--loss', default=-1., help='Job number.', show_default=True)
def report_loss(job,loss):
	with sqlite3.connect('jobs.db', check_same_thread=False) as conn:
		pd.DataFrame([job,val_loss],index=['job','val_loss'],columns=[0]).T.to_sql('val_loss',conn,if_exists='append')

if __name__=='__main__':
	methylcaps()
