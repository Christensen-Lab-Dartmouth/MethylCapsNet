from threading import Thread
#import dask
import chocolate as choco
import time
import numpy as np, pandas as pd
import subprocess
import sqlite3
import click
from submit_hpc.job_generator import assemble_run_torque
#from dask.distributed import Client, as_completed
RANDOM_SEED=42
np.random.seed(RANDOM_SEED)


CONTEXT_SETTINGS = dict(help_option_names=['-h','--help'], max_content_width=90)

@click.group(context_settings= CONTEXT_SETTINGS)
@click.version_option(version='0.1')
def hypscan():
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
		pass
		#print('read jobs')
		# with sqlite3.connect('jobs.db', check_same_thread=False) as conn:
		# 	# c=conn.cursor()
		# 	# c.execute("""SELECT count(name) FROM sqlite_master WHERE type='table' AND name='val_loss';""")
		# 	# if c.fetchone()[0]==1:
		# 	try:
		# 		df=pd.read_sql("select * from 'val_loss'",conn).set_index('job')
		# 		#print(self.job in list(df.index))
		# 		if self.job in list(df.index):
		# 			self.val_loss=df.loc[self.job,'val_loss']
		# 		else:
		# 			self.val_loss=-1
		# 	except Exception as e:
		# 		print(e)
		# 		self.val_loss=-1
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

	if torque:
		assemble_run_torque(command, use_gpu=gpu, additions=additional_command, queue='gpuq' if gpu else "normal", time=np.ceil(total_time/60.), ngpu=1, additional_options=additional_options)
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

@hypscan.command()
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
@click.option('-j', '--n_jobs', default=300, help='Total number jobs to successfully run.', show_default=True)
@click.option('-w', '--n_workers', default=6, help='Total number jobs running at same time.', show_default=True)
@click.option('-u', '--update', is_flag=True, help='Update in script.')
@click.option('-rs', '--random_seed', default=42, help='Random state.')
@click.option('-ot', '--optimize_time', is_flag=True, help='Optimize model for compute time.')
@click.option('-cc', '--capsule_choice', default=['genomic_binned'], multiple=True, help='Specify multiple sets of capsules to include. Cannot specify both custom_bed and custom_set.', show_default=True, type=click.Choice(["GSEA",'all_gene_sets','genomic_binned','custom_bed','custom_set','gene', 'gene_context', 'GSEA_C5.BP', 'GSEA_C6', 'GSEA_C1', 'GSEA_H', 'GSEA_C3.MIR', 'GSEA_C2.CGP', 'GSEA_C4.CM', 'GSEA_C5.CC', 'GSEA_C3.TFT', 'GSEA_C5.MF', 'GSEA_C7', 'GSEA_C2.CP', 'GSEA_C4.CGN', 'UCSC_RefGene_Name', 'UCSC_RefGene_Accession', 'UCSC_RefGene_Group', 'UCSC_CpG_Islands_Name', 'Relation_to_UCSC_CpG_Island', 'Phantom', 'DMR', 'Enhancer', 'HMM_Island', 'Regulatory_Feature_Name', 'Regulatory_Feature_Group', 'DHS']))# ADD LOLA!!!
@click.option('-cf', '--custom_capsule_file', default='', help='Custom capsule file, bed or pickle.', show_default=True, type=click.Path(exists=False))
@click.option('-rt', '--retrain_top_job', is_flag=True,  help='Custom capsule file, bed or pickle.', show_default=True)
@click.option('-bs', '--batch_size', default=16, help='Batch size.', show_default=True)
@click.option('-op', '--output_top_job_params', is_flag=True,  help='Output parameters of top job.', show_default=True)
@click.option('-lc', '--limited_capsule_names_file', default='', help='File of new line delimited names of capsules to filter from larger list.', show_default=True, type=click.Path(exists=False))
@click.option('-ne', '--n_epochs', default=10, help='Number of epochs. Setting to 0 induces scan of epochs.')
@click.option('-mcl', '--min_capsule_len_low_bound', default=50, help='Low bound of min number in capsules.', show_default=True)
@click.option('-gsea', '--gsea_superset', default='', help='GSEA supersets.', show_default=True, type=click.Choice(['','C1', 'C3.MIR', 'C3.TFT', 'C7', 'C5.MF', 'H', 'C5.BP', 'C2.CP', 'C2.CGP', 'C4.CGN', 'C5.CC', 'C6', 'C4.CM']))
@click.option('-ts', '--tissue', default='', help='Tissue associated with GSEA.', show_default=True, type=click.Choice(['','adipose tissue','adrenal gland','appendix','bone marrow','breast','cerebral cortex','cervix, uterine','colon','duodenum','endometrium','epididymis','esophagus','fallopian tube','gallbladder','heart muscle','kidney','liver','lung','lymph node','ovary','pancreas','parathyroid gland','placenta','prostate','rectum','salivary gland','seminal vesicle','skeletal muscle','skin','small intestine','smooth muscle','spleen','stomach','testis','thyroid gland','tonsil','urinary bladder','ubiquitous']))
@click.option('-ns', '--number_sets', default=25, help='Number top gene sets to choose for tissue-specific gene sets.', show_default=True)
@click.option('-st', '--use_set', is_flag=True, help='Use sets or genes within sets.', show_default=True)
@click.option('-gc', '--gene_context', is_flag=True, help='Use upstream and gene body contexts for gsea analysis.', show_default=True)
@click.option('-ss', '--select_subtypes', default=[''], multiple=True, help='Selected subtypes if looking to reduce number of labels to predict', show_default=True)
@click.option('-hyp', '--custom_hyperparameters', default='hyperparameters.yaml', help='Custom hyperparameter yaml file, bed or pickle.', show_default=True, type=click.Path(exists=False))
@click.option('-mc', '--min_capsules', default=5, help='Minimum number of capsules in analysis.', show_default=True)
@click.option('-fp', '--fit_spw', is_flag=True, help='Fit SPWNet for feature selection.', show_default=True)
@click.option('-l1l2', '--l1_l2', default='', help='L1, L2 penalization, comma delimited.', type=click.Path(exists=False), show_default=True)
def hyperparameter_scan(train_methyl_array,
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
						n_jobs,
						n_workers,
						update,
						random_seed,
						optimize_time,
						capsule_choice,
						custom_capsule_file,
						retrain_top_job,
						batch_size,
						output_top_job_params,
						limited_capsule_names_file,
						n_epochs,
						min_capsule_len_low_bound,
						gsea_superset,
						tissue,
						number_sets,
						use_set,
						gene_context,
						select_subtypes,
						custom_hyperparameters,
						min_capsules,
						fit_spw,
						l1_l2):

	np.random.seed(random_seed)

	#subprocess.call('rm -f jobs.db',shell=True)

	opts=dict(train_methyl_array=train_methyl_array,
							val_methyl_array=val_methyl_array,
							interest_col=interest_col,
							n_bins=n_bins,
							custom_loss=custom_loss,
							search_strategy=search_strategy,
							total_time=total_time,
							delay_time=delay_time,
							random_state=random_seed,
							batch_size=batch_size,
							n_epochs=n_epochs,
							min_capsule_len_low_bound=min_capsule_len_low_bound,
							number_sets=number_sets,
							custom_hyperparameters=custom_hyperparameters,
							min_capsules=min_capsules)
	if torque and not update:
		opts['torque']=''
	if use_set:
		opts['use_set']=''
	if gene_context:
		opts['gene_context']=''
	if fit_spw:
		opts['fit_spw']=''
	if l1_l2:
		opts['l1_l2']=l1_l2
	if gsea_superset:
		opts['gsea_superset']=gsea_superset
	if tissue:
		opts['tissue']=tissue
	if gpu:
		opts['gpu']=''
	if optimize_time:
		opts['optimize_time']=''
	if capsule_choice:
		opts['capsule_choice']=' -cc '.join(list(filter(None,capsule_choice)))
	select_subtypes=list(filter(None,select_subtypes))
	if select_subtypes:
		opts['select_subtypes']=' -ss '.join(select_subtypes)
	if limited_capsule_names_file:
		opts['limited_capsule_names_file']=limited_capsule_names_file
	if retrain_top_job:
		n_jobs=1
		opts['retrain_top_job']=''
	if output_top_job_params:
		opts['output_top_job_params']=''
	if custom_capsule_file:
		opts['custom_capsule_file']=custom_capsule_file
	additional_opts=dict(additional_command=additional_command,
						additional_options=additional_options)
	for job in [np.random.randint(0,10000000) for i in range(n_jobs)]:
		opts['job']=job
		command='methylcaps-hypjob hyperparameter_job {} {}'.format(' '.join(['--{} {}'.format(k,v) for k,v in opts.items()]),' '.join(['--{} "{}"'.format(k,v) for k,v in additional_opts.items()]))
		if update:
			command='{} {}'.format(command,'-u')
		command='{} {}'.format(command,'&' if not (torque and update) else '')
		if update:
			if torque:
				assemble_run_torque(command, use_gpu=gpu, queue='gpuq' if gpu else "normal", time=int(np.ceil(total_time/60.)), ngpu=1, additions=additional_opts['additional_command'],additional_options=additional_opts['additional_options'])
			else:
				command='{} {}'.format('CUDA_VISIBLE_DEVICES=0' if gpu else '',command)
		else:
			subprocess.call(command,shell=True)
	# additional_params=dict(train_methyl_array=train_methyl_array,
	# 						val_methyl_array=val_methyl_array,
	# 						interest_col=interest_col,
	# 						n_bins=n_bins,
	# 						custom_loss=custom_loss)
	#
	#
	# def score_loss(args):
	# 	params,i=args
	#
	# 	job=np.random.randint(0,1000000)
	#
	# 	# with sqlite3.connect('jobs.db', check_same_thread=False) as conn:
	# 	# 	# c=conn.cursor()
	# 	# 	# c.execute("""SELECT count(name) FROM sqlite_master WHERE type='table' AND name='jobs';""")
	# 	# 	# if c.fetchone()[0]==1:
	# 	# 	# 	job=pd.read_sql("select * from 'jobs'",conn)['job'].max()+1
	# 	# 	# else:
	# 	# 	# 	job=i+1
	# 	# 	try:
	# 	# 		job=pd.read_sql("select * from 'jobs'",conn)['job'].max()+1
	# 	# 	except:
	# 	# 		job=i+1
	# 	# 	# del c
	# 	#
	# 	# with sqlite3.connect('jobs.db', check_same_thread=False) as conn:
	# 	# 	pd.DataFrame([job],index=[0],columns=['job']).to_sql('jobs',conn,if_exists='append')
	#
	# 	params['hidden_topology']=','.join([str(params['encoder_layer_{}_size'.format(j)]) for j in range(params['num_encoder_hidden_layers'])])
	# 	params['decoder_topology']=','.join([str(params['decoder_layer_{}_size'.format(j)]) for j in range(params['num_decoder_hidden_layers'])])
	#
	# 	# for j in range(params['num_encoder_hidden_layers']):
	# 	# 	del params['encoder_layer_{}_size'.format(j)]
	# 	#
	# 	# for j in range(params['num_decoder_hidden_layers']):
	# 	# 	del params['decoder_layer_{}_size'.format(j)]
	#
	# 	for k in list(params.keys()):
	# 		if k.endswith('_size'):
	# 			del params[k]
	#
	# 	del params['num_encoder_hidden_layers'], params['num_decoder_hidden_layers']
	#
	# 	params.update(additional_params)
	#
	# 	params['job']=job
	#
	# 	print(params)
	#
	# 	command='{} python methylcapsnet_cli.py train_capsnet {} || python methylcapsnet_cli.py report_loss -j {}'.format('CUDA_VISIBLE_DEVICES=0' if gpu and not torque else '',' '.join(['--{} {}'.format(k,v) for k,v in params.items() if v]),params['job'])#,'&' if not torque else '')
	#
	# 	val_loss = return_val_loss(command, torque, total_time, delay_time, job, gpu, additional_command, additional_options)
	#
	# 	return val_loss
	#
	# def return_loss(args):
	# 	token,args=args
	# 	#print(token)
	# 	return token, score_loss(args)
	#
	# grid=dict(n_epochs=choco.quantized_uniform(low=10, high=50, step=10),
	# 			bin_len=choco.quantized_uniform(low=100000, high=1000000, step=100000),
	# 			min_capsule_len=choco.quantized_uniform(low=50, high=500, step=50),
	# 			primary_caps_out_len=choco.quantized_uniform(low=10, high=100, step=5),
	# 			caps_out_len=choco.quantized_uniform(low=10, high=100, step=5),
	# 			num_encoder_hidden_layers={i: {'encoder_layer_{}_size'.format(j):choco.quantized_uniform(10,100,10) for j in range(i+1)} for i in range(3)},
	# 			#hidden_topology=,
	# 			gamma=choco.quantized_log(-5,-1,1,10),
	# 			num_decoder_hidden_layers={i: {'decoder_layer_{}_size'.format(j):choco.quantized_uniform(10,100,10) for j in range(i+1)} for i in range(3)},
	# 			#decoder_topology=,
	# 			learning_rate=choco.quantized_log(-5,-1,1,10),
	# 			routing_iterations=choco.quantized_uniform(low=2, high=6, step=1),
	# 			overlap=choco.quantized_uniform(low=0., high=.9, step=.1),
	# 			gamma2=choco.quantized_log(-5,-1,1,10)
	# 		)
	#
	# optimization_method = 'bayes'
	# optimization_methods=['random','quasi','bayes']
	#
	# sampler_opts={}
	#
	# if optimization_method in ['random']:
	# 	sampler_opts['random_state']=42
	# elif optimization_method in ['quasi']:
	# 	sampler_opts['seed']=42
	# 	sampler_opts['skip']=3
	# elif optimization_method in ['bayes']:
	# 	sampler_opts['n_bootstrap']=10
	# 	#sampler_opts['random_state']=42
	#
	# optimizer = dict(random=choco.Random,quasi=choco.QuasiRandom,bayes=choco.Bayes)[optimization_method]
	#
	# hyp_conn = choco.SQLiteConnection(url="sqlite:///hyperparameter_scan.db")
	#
	# sampler = optimizer(hyp_conn, grid, **sampler_opts)
	#
	# in_batches=False
	# if in_batches:
	# 	for j in range(n_jobs//n_workers): # add a continuous queue in the future
	# 		token_loss_list = dask.compute(*[(dask.delayed(lambda x: x)(token),dask.delayed(score_loss)((params,i))) for i,(token,params) in enumerate([sampler.next() for i in range(n_workers)])],scheduler='processes',num_workers=n_workers)
	# 		for token,loss in token_loss_list:
	# 			if loss>=0:
	# 				sampler.update(token, loss)
	# else:
	# 	client = Client(processes=True)
	# 	token_loss_list=client.map(return_loss,[(token,(params,i)) for i,(token,params) in enumerate([sampler.next() for i in range(n_workers)])])
	# 	pool=as_completed(token_loss_list)
	# 	i=n_workers
	# 	for future in pool:
	# 		token,val_loss=future.result()
	# 		if val_loss>=0:
	# 			sampler.update(token, loss)
	# 		token,params=sampler.next()
	# 		new_future=client.submit(return_loss,(token,(params,i)))
	# 		pool.add(new_future)
	# 		i+=1
	# 		if i > n_jobs:
	# 			break
	#
	# 	client.close()


if __name__=='__main__':
	hypscan()
