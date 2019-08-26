from threading import Thread
import dask
import chocolate as choco
import time
import numpy as np, pandas as pd
import subprocess
import sqlite3
import click
RANDOM_SEED=42
np.random.seed(42)


CONTEXT_SETTINGS = dict(help_option_names=['-h','--help'], max_content_width=90)

@click.group(context_settings= CONTEXT_SETTINGS)
@click.version_option(version='0.1')
def hypscan():
	pass

class MonitorJobs(Thread):
	def __init__(self, start_time, delay, end_time, job):
		super(Monitor, self).__init__()
		self.stopped = False
		self.start_time = start_time
		self.end_time = end_time
		self.delay = delay # Time between calls to GPUtil
		self.val_loss = -1
		self.job = job
		self.start()

	def search_jobs(self):
		conn = sqlite3.connect('jobs.db')
		df=pd.read_sql("select * from 'val_loss'",conn).set_index('job')
		conn.close()
		if self.job in list(df.index):
			self.val_loss=df.loc[self.job,'val_loss']
		else:
			self.val_loss=-1

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
		assemble_run_torque(command, use_gpu=gpu, additions=additional_command, queue='gpuq' if cuda else "normal", time=np.ceil(total_time/60.), ngpu=1, additional_options=additional_options)
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
@click.option('-gpu', '--gpu', default=-1, help='If torque submit, which gpu to use.', show_default=True)
@click.option('-a', '--additional_command', default='', help='Additional command to input for torque run.', type=click.Path(exists=False))
@click.option('-ao', '--additional_options', default='', help='Additional options to input for torque run.', type=click.Path(exists=False))
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
						additional_options):


	additional_params=dict(train_methyl_array=train_methyl_array,
							val_methyl_array=val_methyl_array,
							interest_col=interest_col,
							n_bins=n_bins,
							custom_loss=custom_loss)


	def score_loss(params):
		conn = sqlite3.connect('jobs.db')
		df=pd.read_sql("select * from 'jobs'",conn)
		conn.close()
		job=df['job'].max()+1

		conn = sqlite3.connect('jobs.db')
		pd.DataFrame([job],index=[0],columns=['job']).to_sql('jobs',conn,if_exists='append')
		conn.close()

		params['hlt']=','.join([str(params['encoder_layer_{}_size'.format(j)]) for j in range(params['num_encoder_hidden_layers'])])
		params['dlt']=','.join([str(params['decoder_layer_{}_size'.format(j)]) for j in range(params['num_decoder_hidden_layers'])])
		del params['num_encoder_hidden_layers'], params['num_decoder_hidden_layers']

		for j in range(params['num_encoder_hidden_layers']):
			del params['encoder_layer_{}_size'.format(j)]

		for j in range(params['num_decoder_hidden_layers']):
			del params['decoder_layer_{}_size'.format(j)]

		params.update(additional_params)

		command='{} python methylcapsnet_cli.py train_capsnet {}'.format('CUDA_VISIBLE_DEVICES=0' if gpu and not torque else '',' '.join(['-{} {}'.format(k,v) for k,v in params.items()]))

		val_loss = return_val_loss(command, torque, total_time, delay_time, job, gpu, additional_command, additional_options)

		return val_loss

	conn = choco.SQLiteConnection(url="sqlite:///hyperparameter_scan.db")



	grid=dict(n_epochs=choco.quantized_uniform(low=10, high=50, step=10),
				bin_len=choco.quantized_uniform(low=100000, high=1000000, step=100000),
				min_capsule_len=choco.quantized_uniform(low=50, high=500, step=50),
				primary_caps_out_len=choco.quantized_uniform(low=10, high=100, step=5),
				caps_out_len=choco.quantized_uniform(low=10, high=100, step=5),
				num_encoder_hidden_layers={i: {'encoder_layer_{}_size'.format(j):choco.quantized_uniform(10,100,10) for j in range(i+1)} for i in range(3)},
				#hidden_topology=,
				gamma=choco.quantized_log(-5,-1,1,10),
				num_decoder_hidden_layers={i: {'decoder_layer_{}_size'.format(j):choco.quantized_uniform(10,100,10) for j in range(i+1)} for i in range(3)},
				#decoder_topology=,
				learning_rate=choco.quantized_log(-5,-1,1,10),
				routing_iterations=choco.quantized_uniform(low=2, high=6, step=1),
				overlap=choco.quantized_uniform(low=0., high=.9, step=.1),
				gamma2=choco.quantized_log(-5,-1,1,10)
			)

	optimization_method = 'bayes'
	optimization_methods=['random','quasi','bayes']

	sampler_opts={}

	if optimization_method in ['random']:
		sampler_opts['random_state']=42
	elif optimization_method in ['quasi']:
		sampler_opts['seed']=42
		sampler_opts['skip']=3
	elif optimization_method in ['bayes']:
		sampler_opts['n_bootstrap']=10
		sampler_opts['random_state']=42

	optimizer = dict(random=choco.Random,quasi=choco.QuasiRandom,bayes=choco.Bayes)[optimization_method]

	sampler = optimizer(conn, grid, **sampler_opts)

	n_jobs=300
	n_workers=10
	for i in range(n_jobs//n_workers): # add a continuous queue in the future
		token_loss_list = dask.compute(*[(dask.delayed(lambda x: x)(token),dask.delayed(score_loss)(params)) for token,params in [sampler.next() for i in range(n_workers)]],scheduler='processes',num_workers=n_workers)
		for token,loss in token_loss_list:
			if loss!=-1:
				sampler.update(token, loss)

	conn.close()

if __name__=='__main__':
	hypscan()
