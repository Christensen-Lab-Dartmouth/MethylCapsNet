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
from methylcaps_train_ import *
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
def train_capsnet(train_methyl_array,
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

	train_capsnet_(train_methyl_array,
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
						job)


@methylcaps.command()
@click.option('-j', '--job', default=0, help='Job number.', show_default=True)
@click.option('-l', '--loss', default=-1., help='Job number.', show_default=True)
def report_loss(job,loss):
	with sqlite3.connect('jobs.db', check_same_thread=False) as conn:
		pd.DataFrame([job,val_loss],index=['job','val_loss'],columns=[0]).T.to_sql('val_loss',conn,if_exists='append')

if __name__=='__main__':
	methylcaps()
