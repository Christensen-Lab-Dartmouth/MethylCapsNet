from threading import Thread
#import dask
import chocolate as choco
import time
import numpy as np, pandas as pd
import subprocess
import sqlite3
import click
from methylnet.torque_jobs import assemble_run_torque
import time
import pysnooper
#from dask.distributed import Client, as_completed
from methylcapsnet.methylcaps_model_ import *

CONTEXT_SETTINGS = dict(help_option_names=['-h','--help'], max_content_width=90)

@click.group(context_settings= CONTEXT_SETTINGS)
@click.version_option(version='0.1')
def hypjob():
	pass

class MonitorJobs(Thread):
	def __init__(self, start_time, delay, end_time, job):
		super(MonitorJobs, self).__init__()
		self.stopped = False
		self.start_time = start_time
		self.end_time = end_time
		self.delay = delay # Time between calls to GPUtil
		self.val_loss = -1
		self.job = job
		self.start()

	def search_jobs(self):
		#print('read jobs')
		with sqlite3.connect('jobs.db', check_same_thread=False) as conn:
			# c=conn.cursor()
			# c.execute("""SELECT count(name) FROM sqlite_master WHERE type='table' AND name='val_loss';""")
			# if c.fetchone()[0]==1:
			try:
				df=pd.read_sql("select * from 'val_loss'",conn).set_index('job')
				#print(self.job in list(df.index))
				if self.job in list(df.index):
					self.val_loss=df.loc[self.job,'val_loss']
				else:
					self.val_loss=-1
			except Exception as e:
				#print(e)
				self.val_loss=-1
		# else:
		# 	self.val_loss=-1
		# del c
		# try:
		# 	df=pd.read_pickle('jobs.p').set_index('job')
		# 	if self.job in list(df.index):
		# 		self.val_loss=df.loc[self.job,'val_loss']
		# 	else:
		# 		self.val_loss=-1
		# except:
		# 	self.val_loss=-1

	def run(self):
		time_from_start = 0.
		while time_from_start <= self.end_time and self.val_loss==-1:
			self.search_jobs()
			time.sleep(self.delay)
		self.stop()

	def stop(self):
		self.stopped = True

	def return_val_loss(self):
		return self.val_loss

def return_val_loss(command, torque, total_time, delay_time, job, gpu, additional_command, additional_options):
	print(int(np.ceil(total_time/60.)))
	if torque:
		assemble_run_torque(command, use_gpu=gpu, additions=additional_command, queue='gpuq' if gpu else "normal", time=int(np.ceil(total_time/60.)), ngpu=1, additional_options=additional_options)
	else:
		subprocess.call(command,shell=True)

	total_time*= 60.
	start_time = time.time()

	monitor = MonitorJobs(start_time, delay_time, total_time, job=job)

	monitor.run()

	while not monitor.stopped:
		time.sleep(delay_time)

	val_loss = monitor.return_val_loss()

	return val_loss

#@pysnooper.snoop('hypjob.log')
def hyperparameter_job_(train_methyl_array,
						val_methyl_array,
						interest_col,
						n_bins,
						custom_loss,
						torque,
						search_strategy,
						total_time,
						delay_time,
						gpu,
						additional_command,
						additional_options,
						update,
						n_epochs,
						job,
						survival,
						optimize_time,
						random_state,
						capsule_choice,
						custom_capsule_file,
						retrain_top_job,
						batch_size,
						output_top_job_params,
						limited_capsule_names_file,
						min_capsule_len_low_bound,
						gsea_superset,
						tissue,
						number_sets,
						use_set,
						gene_context,
						select_subtypes,
						custom_hyperparameters):

	additional_params=dict(train_methyl_array=train_methyl_array,
							val_methyl_array=val_methyl_array,
							interest_col=interest_col,
							n_bins=n_bins,
							custom_loss=custom_loss,
							job=job,
							batch_size=batch_size,
							number_sets=number_sets,
							)

	if n_epochs:
		additional_params['n_epochs']=n_epochs

	if gsea_superset:
		additional_params['gsea_superset']=gsea_superset
	if tissue:
		additional_params['tissue']=tissue

	if custom_capsule_file:
		additional_params['custom_capsule_file']=custom_capsule_file

	if output_top_job_params:
		retrain_top_job=True

	if limited_capsule_names_file:
		additional_params['limited_capsule_names_file']=limited_capsule_names_file

	if update and not (retrain_top_job and output_top_job_params):
		additional_params['capsule_choice']=capsule_choice
		select_subtypes=list(filter(None,select_subtypes))
		if select_subtypes:
			additional_params['select_subtypes']=select_subtypes
		if use_set:
			additional_params['use_set']=use_set
		if gene_context:
			additional_params['gene_context']=gene_context
	else:
		select_subtypes=list(filter(None,select_subtypes))
		if select_subtypes:
			additional_params['select_subtypes']=' -ss '.join(list(filter(None,select_subtypes)))
		additional_params['capsule_choice']=' -cc '.join(list(filter(None,capsule_choice)))
		if use_set:
			additional_params['use_set']=''
		if gene_context:
			additional_params['gene_context']=''

	if not survival:
		additional_params['gamma2']=1e-2

	def score_loss(params):
		#job=np.random.randint(0,1000000)
		start_time=time.time()

		params['hidden_topology']=','.join([str(int(params['el{}s'.format(j)])) for j in range(params['nehl']+1)])
		params['decoder_topology']=','.join([str(int(params['dl{}s'.format(j)])) for j in range(params['ndhl']+1)])

		del_params=['el{}s'.format(j) for j in range(params['nehl']+1)]+['dl{}s'.format(j) for j in range(params['ndhl']+1)]

		del_params=set(del_params+[k for k in params if k.startswith('el') or k.startswith('dl')])
		# for k in list(params.keys()):
		# 	if k.endswith('_size'):
		# 		del params[k]
		# print(params)
		# print(params['nehl'],params['ndhl'])
		# print(del_params)
		for param in del_params:
			del params[param]

		del params['nehl'], params['ndhl']

		params.update(additional_params)

		print(params)

		command='{} methylcaps-model model_capsnet {} || methylcaps-model report_loss -j {}'.format('CUDA_VISIBLE_DEVICES=0' if gpu and not torque else '',' '.join(['--{} {}'.format(k,v) for k,v in params.items() if v or k=='use_set']),params['job'])#,'&' if not torque else '')

		if output_top_job_params and retrain_top_job:
			print('Top params command: ')
			print('{} --predict'.format(command.split('||')[0]))
			exit()
		elif output_top_job_params:
			print('Continuing training of random parameters, please specify retrain_top_job.')

		if update:

			val_loss = model_capsnet_(**params)

		else:

			val_loss = return_val_loss(command, torque, total_time, delay_time, job, gpu, additional_command, additional_options)

		end_time=time.time()

		if optimize_time:
			return val_loss, start_time-end_time
		else:
			return val_loss

	grid=dict(n_epochs=dict(low=10, high=50, step=10),
				bin_len=dict(low=500000, high=1000000, step=100000),
				min_capsule_len=dict(low=min_capsule_len_low_bound, high=500, step=25),
				primary_caps_out_len=dict(low=10, high=100, step=5),
				caps_out_len=dict(low=10, high=100, step=5),
				nehl=dict(low=10,high=300,step=10,n_layers=3),
				ndhl=dict(low=100,high=300,step=10,n_layers=3),
				learning_rate=dict(low=-5,high=-1,step=1,base=10),
				gamma=dict(low=-5,high=-1,step=1,base=10),
				gamma2=dict(low=-5,high=-1,step=1,base=10),
				overlap=dict(low=0., high=.5, step=.1),
				routing_iterations=dict(low=2, high=4, step=1))

	if os.path.exists(custom_hyperparameters):
		from ruamel.yaml import safe_load as load
		with open(custom_hyperparameters) as f:
			new_grid = load(f)
		print(new_grid)
		for k in new_grid:
			for k2 in new_grid[k]:
				grid[k][k2]=new_grid[k][k2]


	n_layers=dict(encoder=grid['nehl'].pop('n_layers'),decoder=grid['ndhl'].pop('n_layers'))


	grid=dict(n_epochs=choco.quantized_uniform(**grid['n_epochs']),
				bin_len=choco.quantized_uniform(**grid['bin_len']),
				min_capsule_len=choco.quantized_uniform(**grid['min_capsule_len']),
				primary_caps_out_len=choco.quantized_uniform(**grid['primary_caps_out_len']),
				caps_out_len=choco.quantized_uniform(**grid['caps_out_len']),
				nehl={i: {'el{}s'.format(j):choco.quantized_uniform(**grid['nehl']) for j in range(i+1)} for i in range(n_layers['encoder'])},
				gamma=choco.quantized_log(**grid['gamma']),
				ndhl={i: {'dl{}s'.format(j):choco.quantized_uniform(**grid['ndhl']) for j in range(i+1)} for i in range(n_layers['decoder'])},
				learning_rate=choco.quantized_log(**grid['learning_rate']),
				routing_iterations=choco.quantized_uniform(**grid['routing_iterations']),
				overlap=choco.quantized_uniform(**grid['overlap']),
				gamma2=choco.quantized_log(**grid['gamma2'])
			) # ADD BATCH SIZE

	if n_epochs:
		grid.pop('n_epochs')

	if not survival:
		grid.pop('gamma2')

	if 'genomic_binned' not in list(capsule_choice):
		for k in ['overlap','bin_len']:
			grid.pop(k)

	if retrain_top_job:

		conn=choco.SQLiteConnection('sqlite:///hyperparameter_scan.db')
		results=conn.results_as_dataframe()
		results=results[~results['_loss'].isnull()]
		params=dict(results.iloc[np.argmin(results['_loss'].values)])
		for k in ['bin_len','caps_out_len','min_capsule_len','ndhl','nehl','primary_caps_out_len','routing_iterations']:
			if k in params:
				params[k]=int(params[k])

		del params['_loss']

		top_loss=score_loss(params)

		pickle.dump(top_loss,open('top_loss.pkl','wb'))

	else:

		optimization_method = search_strategy#'bayes'
		optimization_methods=['random','quasi','bayes']

		sampler_opts={}

		if optimization_method in ['random']:
			sampler_opts['n_bootstrap']=10000
			#sampler_opts['random_state']=random_state
		elif optimization_method in ['quasi']:
			sampler_opts['seed']=random_state
			sampler_opts['skip']=3
		elif optimization_method in ['bayes']:
			sampler_opts['n_bootstrap']=35
			sampler_opts['utility_function']='ei'
			sampler_opts['xi']=0.1
			#sampler_opts['random_state']=42

		#print(optimization_method)
		optimizer = dict(random=choco.Bayes,quasi=choco.QuasiRandom,bayes=choco.Bayes)[optimization_method] # Random

		hyp_conn = choco.SQLiteConnection(url="sqlite:///hyperparameter_scan.db")

		sampler = optimizer(hyp_conn, grid, **sampler_opts)

		#print(sampler)

		if 0 and optimization_method in ['bayes']:
			sampler.random_state=np.random.RandomState(42)

		token,params=sampler.next()

		loss=score_loss(params)

		if (loss if not optimize_time else loss[0])>=0:
			sampler.update(token, loss)


@hypjob.command()
@click.option('-i', '--train_methyl_array', default='./train_val_test_sets/train_methyl_array.pkl', help='Input database for beta and phenotype data.', type=click.Path(exists=False), show_default=True)
@click.option('-v', '--val_methyl_array', default='./train_val_test_sets/val_methyl_array.pkl', help='Test database for beta and phenotype data.', type=click.Path(exists=False), show_default=True)
@click.option('-ic', '--interest_col', default='disease', help='Specify column looking to make predictions on.', show_default=True)
@click.option('-nb', '--n_bins', default=0, help='Number of bins if column is continuous variable.', show_default=True)
@click.option('-cl', '--custom_loss', default='none', help='Specify custom loss function.', show_default=True, type=click.Choice(['none','cox']))
@click.option('-t', '--torque', is_flag=True, help='Submit jobs on torque.')
@click.option('-s', '--search_strategy', default='bayes', help='Search strategy.', type=click.Choice(['bayes','random','quasi']))
@click.option('-tt', '--total_time', default=60, help='Total time to run each job in minutes.', show_default=True)
@click.option('-dt', '--delay_time', default=60, help='Total time to wait before searching for output job in seconds.', show_default=True)
@click.option('-gpu', '--gpu', is_flag=True, help='If torque submit, which gpu to use.')
@click.option('-a', '--additional_command', default='', help='Additional command to input for torque run.', type=click.Path(exists=False))
@click.option('-ao', '--additional_options', default='', help='Additional options to input for torque run.', type=click.Path(exists=False))
@click.option('-u', '--update', is_flag=True, help='Update in script.')
@click.option('-ne', '--n_epochs', default=10, help='Number of epochs. Setting to 0 induces scan of epochs.')
@click.option('-j', '--job', default=42, help='Job name.')
@click.option('-srv', '--survival', is_flag=True, help='Scan parameters for survival analysis.')
@click.option('-ot', '--optimize_time', is_flag=True, help='Optimize model for compute time.')
@click.option('-rs', '--random_state', default=42, help='Random state.')
@click.option('-cc', '--capsule_choice', default=['genomic_binned'], multiple=True, help='Specify multiple sets of capsules to include. Cannot specify both custom_bed and custom_set.', show_default=True, type=click.Choice(["GSEA",'all_gene_sets','genomic_binned','custom_bed','custom_set','gene', 'gene_context', 'GSEA_C5.BP', 'GSEA_C6', 'GSEA_C1', 'GSEA_H', 'GSEA_C3.MIR', 'GSEA_C2.CGP', 'GSEA_C4.CM', 'GSEA_C5.CC', 'GSEA_C3.TFT', 'GSEA_C5.MF', 'GSEA_C7', 'GSEA_C2.CP', 'GSEA_C4.CGN', 'UCSC_RefGene_Name', 'UCSC_RefGene_Accession', 'UCSC_RefGene_Group', 'UCSC_CpG_Islands_Name', 'Relation_to_UCSC_CpG_Island', 'Phantom', 'DMR', 'Enhancer', 'HMM_Island', 'Regulatory_Feature_Name', 'Regulatory_Feature_Group', 'DHS']))# ADD LOLA!!!
@click.option('-cf', '--custom_capsule_file', default='', help='Custom capsule file, bed or pickle.', show_default=True, type=click.Path(exists=False))
@click.option('-rt', '--retrain_top_job', is_flag=True,  help='Retrain top job.', show_default=True)
@click.option('-bs', '--batch_size', default=16, help='Batch size.', show_default=True)
@click.option('-op', '--output_top_job_params', is_flag=True,  help='Output parameters of top job.', show_default=True)
@click.option('-lc', '--limited_capsule_names_file', default='', help='File of new line delimited names of capsules to filter from larger list.', show_default=True, type=click.Path(exists=False))
@click.option('-mcl', '--min_capsule_len_low_bound', default=50, help='Low bound of min number in capsules.', show_default=True)
@click.option('-gsea', '--gsea_superset', default='', help='GSEA supersets.', show_default=True, type=click.Choice(['','C1', 'C3.MIR', 'C3.TFT', 'C7', 'C5.MF', 'H', 'C5.BP', 'C2.CP', 'C2.CGP', 'C4.CGN', 'C5.CC', 'C6', 'C4.CM']))
@click.option('-ts', '--tissue', default='', help='Tissue associated with GSEA.', show_default=True, type=click.Choice(['','adipose tissue','adrenal gland','appendix','bone marrow','breast','cerebral cortex','cervix, uterine','colon','duodenum','endometrium','epididymis','esophagus','fallopian tube','gallbladder','heart muscle','kidney','liver','lung','lymph node','ovary','pancreas','parathyroid gland','placenta','prostate','rectum','salivary gland','seminal vesicle','skeletal muscle','skin','small intestine','smooth muscle','spleen','stomach','testis','thyroid gland','tonsil','urinary bladder','ubiquitous']))
@click.option('-ns', '--number_sets', default=25, help='Number top gene sets to choose for tissue-specific gene sets.', show_default=True)
@click.option('-st', '--use_set', is_flag=True, help='Use sets or genes within sets.', show_default=True)
@click.option('-gc', '--gene_context', is_flag=True, help='Use upstream and gene body contexts for gsea analysis.', show_default=True)
@click.option('-ss', '--select_subtypes', default=[''], multiple=True, help='Selected subtypes if looking to reduce number of labels to predict', show_default=True)
@click.option('-hyp', '--custom_hyperparameters', default='hyperparameters.yaml', help='Custom hyperparameter yaml file, bed or pickle.', show_default=True, type=click.Path(exists=False))
def hyperparameter_job(train_methyl_array,
						val_methyl_array,
						interest_col,
						n_bins,
						custom_loss,
						torque,
						search_strategy,
						total_time,
						delay_time,
						gpu,
						additional_command,
						additional_options,
						update,
						n_epochs,
						job,
						survival,
						optimize_time,
						random_state,
						capsule_choice,
						custom_capsule_file,
						retrain_top_job,
						batch_size,
						output_top_job_params,
						limited_capsule_names_file,
						min_capsule_len_low_bound,
						gsea_superset,
						tissue,
						number_sets,
						use_set,
						gene_context,
						select_subtypes,
						custom_hyperparameters):

	hyperparameter_job_(train_methyl_array,
							val_methyl_array,
							interest_col,
							n_bins,
							custom_loss,
							torque,
							search_strategy,
							total_time,
							delay_time,
							gpu,
							additional_command,
							additional_options,
							update,
							n_epochs,
							job,
							survival,
							optimize_time,
							random_state,
							capsule_choice,
							custom_capsule_file,
							retrain_top_job,
							batch_size,
							output_top_job_params,
							limited_capsule_names_file,
							min_capsule_len_low_bound,
							gsea_superset,
							tissue,
							number_sets,
							use_set,
							gene_context,
							select_subtypes,
							custom_hyperparameters)



if __name__=='__main__':
	hypscan()
