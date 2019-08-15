import pandas as pd
from pymethylprocess.MethylationDataTypes import MethylationArray
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")
from pybedtools import BedTool
import numpy as np
from functools import reduce
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import Adam
from sklearn.preprocessing import LabelBinarizer
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import os
import pysnooper
from sklearn.decomposition import PCA
import plotly.offline as py
import plotly.express as px
import argparse
import pickle
from sklearn.metrics import classification_report

@pysnooper.snoop('main_snoop.log')
def main():
	p=argparse.ArgumentParser()
	p.add_argument('--interest_col',type=str)
	p.add_argument('--n_bins',type=int)
	args=p.parse_args()
	bin_len=1000000
	min_capsule_len=350
	interest_col=args.interest_col
	n_bins=args.n_bins

	primary_caps_out_len=40
	caps_out_len=20
	n_epochs=500
	hidden_topology=[30,80,50]
	gamma=1e-2
	decoder_top=[100, 300]
	lr=1e-3
	routing_iterations=3

	if not os.path.exists('hg19.{}.bed'.format(bin_len)):
		BedTool('hg19.genome').makewindows(g='hg19.genome',w=bin_len).saveas('hg19.{}.bed'.format(bin_len))#.to_dataframe().shape

	ma=MethylationArray.from_pickle('train_val_test_sets/train_methyl_array.pkl')
	ma_v=MethylationArray.from_pickle('train_val_test_sets/val_methyl_array.pkl')

	include_last=False

	@pysnooper.snoop('get_mod.log')
	def get_final_modules(ma=ma,a='450kannotations.bed',b='lola_vignette_data/activeDHS_universe.bed', include_last=False, min_capsule_len=2000):
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

	final_modules,modulecpgs,module_names=get_final_modules(b='hg19.{}.bed'.format(bin_len),include_last=include_last, min_capsule_len=min_capsule_len)
	print('LEN_MODULES',len(final_modules))

	if not include_last:
		ma.beta=ma.beta.loc[:,modulecpgs]
		ma_v.beta=ma_v.beta.loc[:,modulecpgs]
	# https://github.com/higgsfield/Capsule-Network-Tutorial/blob/master/Capsule%20Network.ipynb

	def softmax(input_tensor, dim=1):
		# transpose input
		transposed_input = input_tensor.transpose(dim, len(input_tensor.size()) - 1)
		# calculate softmax
		softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
		# un-transpose result
		return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input_tensor.size()) - 1)

	class MLP(nn.Module): # add latent space extraction, and spits out csv line of SQL as text for UMAP
		def __init__(self, n_input, hidden_topology, dropout_p, n_outputs=1, binary=False, softmax=False):
			super(MLP,self).__init__()
			self.hidden_topology=hidden_topology
			self.topology = [n_input]+hidden_topology+[n_outputs]
			layers = [nn.Linear(self.topology[i],self.topology[i+1]) for i in range(len(self.topology)-2)]
			for layer in layers:
				torch.nn.init.xavier_uniform_(layer.weight)
			self.layers = [nn.Sequential(layer,nn.ReLU(),nn.Dropout(p=dropout_p)) for layer in layers]
			self.output_layer = nn.Linear(self.topology[-2],self.topology[-1])
			torch.nn.init.xavier_uniform_(self.output_layer.weight)
			if binary:
				output_transform = nn.Sigmoid()
			elif softmax:
				output_transform = nn.Softmax()
			else:
				output_transform = nn.Dropout(p=0.)
			self.layers.append(nn.Sequential(self.output_layer,output_transform))
			self.mlp = nn.Sequential(*self.layers)

		def forward(self, x):
			#print(x.shape)
			return self.mlp(x)

	class MethylationDataset(Dataset):
		def __init__(self, methyl_arr, outcome_col,binarizer=None, modules=[]):
			if binarizer==None:
				binarizer=LabelBinarizer()
				binarizer.fit(methyl_arr.pheno[outcome_col].astype(str).values)
			self.y=binarizer.transform(methyl_arr.pheno[outcome_col].astype(str).values)
			self.y_unique=np.unique(np.argmax(self.y,1))
			self.binarizer=binarizer
			if not modules:
				modules=[list(methyl_arr.beta)]
			self.modules=modules
			self.X=methyl_arr.beta
			self.length=methyl_arr.beta.shape[0]

		def __len__(self):
			return self.length

		def __getitem__(self,i):
			return tuple([torch.FloatTensor(self.X.iloc[i].values)]+[torch.FloatTensor(self.X.iloc[i].loc[module].values) for module in self.modules]+[torch.FloatTensor(self.y[i])])

	class PrimaryCaps(nn.Module):
		def __init__(self,modules,hidden_topology,n_output):
			super(PrimaryCaps, self).__init__()
			self.capsules=nn.ModuleList([MLP(len(module),hidden_topology,0.,n_outputs=n_output) for module in modules])

		def forward(self, x):
			#print(self.capsules)
			u = [self.capsules[i](x[i]) for i in range(len(self.capsules))]
			u = torch.stack(u, dim=1)
			#print(u.size())
			return self.squash(u)

		def squash(self, x):
			squared_norm = (x ** 2).sum(-1, keepdim=True)
			#print('prim_norm',squared_norm.size())
			output_tensor = squared_norm *  x / ((1. + squared_norm) * torch.sqrt(squared_norm))
			#print('z_init',output_tensor.size())
			return output_tensor

		def get_weights(self):
			return list(self.capsules[0].parameters())[0].data#self.state_dict()#[self.capsules[i].state_dict() for i in range(len(self.capsules))]

	class CapsLayer(nn.Module):
		def __init__(self, n_capsules, n_routes, n_input, n_output, routing_iterations=3):
			super(CapsLayer, self).__init__()
			self.n_capsules=n_capsules
			self.num_routes = n_routes
			self.W=nn.Parameter(torch.randn(1, n_routes, n_capsules, n_output, n_input))
			self.routing_iterations=routing_iterations
			self.c_ij=None

		def forward(self,x):
			batch_size = x.size(0)
			x = torch.stack([x] * self.n_capsules, dim=2).unsqueeze(4)

			W = torch.cat([self.W] * batch_size, dim=0)
			#print('affine',W.size(),x.size())
			u_hat = torch.matmul(W, x)
			#print('affine_trans',u_hat.size())

			b_ij = Variable(torch.zeros(1, self.num_routes, self.n_capsules, 1))

			if torch.cuda.is_available():
				b_ij=b_ij.cuda()


			for iteration in range(self.routing_iterations):
				self.c_ij = softmax(b_ij)
				#print(c_ij)
				c_ij = torch.cat([self.c_ij] * batch_size, dim=0).unsqueeze(4)
				#print('coeff',c_ij.size())#[0,:,0,:])#.size())

				s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
				v_j = self.squash(s_j)
				#print('z',v_j.size())

				if iteration < self.routing_iterations - 1:
					a_ij = torch.matmul(u_hat.transpose(3, 4), torch.cat([v_j] * self.num_routes, dim=1))
					b_ij = b_ij + a_ij.squeeze(4).mean(dim=0, keepdim=True)

			return v_j.squeeze(1)

		def return_routing_coef(self):
			return self.c_ij

		def squash(self, x):
			#print(x.size())
			squared_norm = (x ** 2).sum(-1, keepdim=True)
			#print('norm',squared_norm.size())
			output_tensor = squared_norm *  x / ((1. + squared_norm) * torch.sqrt(squared_norm))
			return output_tensor

	class Decoder(nn.Module):
		def __init__(self, n_input, n_output, hidden_topology):
			super(Decoder, self).__init__()
			self.decoder=MLP(n_input,hidden_topology, 0., n_outputs=n_output, binary=True)

		def forward(self, x):
			return self.decoder(x)

	class CapsNet(nn.Module):
		def __init__(self, primary_caps, caps_hidden_layers, caps_output_layer, decoder, lr_balance=0.5, gamma=0.005):
			super(CapsNet, self).__init__()
			self.primary_caps=primary_caps
			self.caps_hidden_layers=caps_hidden_layers
			self.caps_output_layer=caps_output_layer
			self.decoder=decoder
			self.recon_loss_fn = nn.BCELoss()
			self.lr_balance=lr_balance
			self.gamma=gamma

		def forward(self, x_orig, modules_input):
			x=self.primary_caps(modules_input)
			primary_caps_out=x#.view(x.size(0),x.size(1)*x.size(2))
			#print(x.size())
			for layer in self.caps_hidden_layers:
				x=layer(x)

			y_pred=self.caps_output_layer(x)#.squeeze(-1)
			#print(y_pred.shape)

			classes = torch.sqrt((y_pred ** 2).sum(2))
			classes = F.softmax(classes)

			max_length_indices = classes.argmax(dim=1)
			masked = torch.sparse.torch.eye(self.caps_output_layer.n_capsules)
			if torch.cuda.is_available():
				masked=masked.cuda()
			masked = masked.index_select(dim=0, index=max_length_indices.squeeze(1).data)

			embedding = (y_pred * masked[:, :, None, None]).view(y_pred.size(0), -1)

			#print(y_pred.size())
			x_hat=self.decoder(embedding)#.reshape(y_pred.size(0),-1))
			return x_orig, x_hat, y_pred, embedding, primary_caps_out

		def recon_loss(self, x_orig, x_hat):
			return self.recon_loss_fn(x_hat, x_orig)

		def margin_loss(self,x, labels):
			batch_size = x.size(0)

			v_c = torch.sqrt((x**2).sum(dim=2, keepdim=True))

			#print(v_c)

			left = (F.relu(0.9 - v_c)**2).view(batch_size, -1)
			right = (F.relu(v_c - 0.1)**2).view(batch_size, -1)
			#print(left)
			#print(right)
			#print(labels)

			loss = labels * left + self.lr_balance * (1.0 - labels) * right
			#print(loss.shape)
			loss = loss.sum(dim=1).mean()
			return loss

		def calculate_loss(self, x_orig, x_hat, y_pred, y_true):
			margin_loss = self.margin_loss(y_pred, y_true)
			recon_loss = self.gamma*self.recon_loss(x_orig,x_hat)
			loss = margin_loss + recon_loss
			return loss, margin_loss, recon_loss

	if n_bins:
		ma.pheno.loc[:,interest_col],bins=pd.cut(ma.pheno[interest_col],bins=n_bins,retbins=True)
		ma_v.pheno.loc[:,interest_col],bins=pd.cut(ma_v.pheno[interest_col],bins=bins,retbins=True,)

	dataset=MethylationDataset(ma,interest_col,modules=final_modules)
	dataset_v=MethylationDataset(ma_v,interest_col,modules=final_modules)

	dataloader=DataLoader(dataset,batch_size=16,shuffle=True,num_workers=8, drop_last=True)
	dataloader_v=DataLoader(dataset_v,batch_size=16,shuffle=False,num_workers=8, drop_last=False)


	n_inputs=list(map(len,final_modules))
	n_primary=len(final_modules)


	primary_caps = PrimaryCaps(modules=final_modules,hidden_topology=hidden_topology,n_output=primary_caps_out_len)
	hidden_caps = []
	n_out_caps=len(dataset.y_unique)
	output_caps = CapsLayer(n_out_caps,n_primary,primary_caps_out_len,caps_out_len,routing_iterations=routing_iterations)
	decoder = Decoder(n_out_caps*caps_out_len,len(list(ma.beta)),decoder_top)
	capsnet = CapsNet(primary_caps, hidden_caps, output_caps, decoder, gamma=gamma)

	if torch.cuda.is_available():
		capsnet=capsnet.cuda()

	for d in ['figures/embeddings'+x for x in ['','2','3']]:
	    os.makedirs(d,exist_ok=True)
	os.makedirs('results/routing_weights',exist_ok=True)
	# extract all c_ij for all layers across all batches, or just last batch
	optimizer = Adam(capsnet.parameters(),lr)
	scheduler=CosineAnnealingLR(optimizer, T_max=10, eta_min=0, last_epoch=-1)
	for epoch in range(n_epochs):
		print(epoch)
		capsnet.train(True)
		running_loss=0.
		Y={'true':[],'pred':[]}
		for i,batch in enumerate(dataloader):
			x_orig=batch[0]
			#print(x_orig)
			y_true=batch[-1]
			module_x = batch[1:-1]
			if torch.cuda.is_available():
				x_orig=x_orig.cuda()
				y_true=y_true.cuda()
				module_x=[mod.cuda() for mod in module_x]
			x_orig, x_hat, y_pred, embedding, primary_caps_out=capsnet(x_orig,module_x)
			loss,margin_loss,recon_loss=capsnet.calculate_loss(x_orig, x_hat, y_pred, y_true)
			Y['true'].extend(y_true.argmax(1).detach().cpu().numpy().tolist())
			Y['pred'].extend(F.softmax(torch.sqrt((y_pred**2).sum(2))).argmax(1).detach().cpu().numpy().tolist())
			train_loss=margin_loss.item()#print(loss)
			running_loss+=train_loss
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		#print(capsnet.primary_caps.get_weights())
		running_loss/=(i+1)
		print('Epoch {}: Train Loss {}, Train R2: {}, Train MAE: {}'.format(epoch,running_loss,r2_score(Y['true'],Y['pred']), mean_absolute_error(Y['true'],Y['pred'])))
		print(classification_report(Y['true'],Y['pred']))
		scheduler.step()
		capsnet.train(False)
		running_loss=np.zeros((3,)).astype(float)
		Y={'true':[],'pred':[],'embeddings':[],'embeddings2':[],'embeddings3':[],'routing_weights':[]}
		with torch.no_grad():
			for i,batch in enumerate(dataloader_v):
				x_orig=batch[0]
				y_true=batch[-1]
				module_x = batch[1:-1]
				if torch.cuda.is_available():
					x_orig=x_orig.cuda()
					y_true=y_true.cuda()
					module_x=[mod.cuda() for mod in module_x]
				x_orig, x_hat, y_pred, embedding, primary_caps_out=capsnet(x_orig,module_x)
				#print(primary_caps_out.size())
				routing_coefs=capsnet.caps_output_layer.return_routing_coef().detach().cpu().numpy()
				if not i:
					Y['routing_weights']=pd.DataFrame(routing_coefs[0,...,0].T,index=dataset.binarizer.classes_,columns=module_names)
				else:
					Y['routing_weights']+=pd.DataFrame(routing_coefs[0,...,0].T,index=dataset.binarizer.classes_,columns=module_names)
				Y['embeddings3'].append(torch.cat([primary_caps_out[i] for i in range(x_orig.size(0))],dim=0).detach().cpu().numpy())
				primary_caps_out=primary_caps_out.view(primary_caps_out.size(0),primary_caps_out.size(1)*primary_caps_out.size(2))
				Y['embeddings'].append(embedding.detach().cpu().numpy())
				Y['embeddings2'].append(primary_caps_out.detach().cpu().numpy())
				loss,margin_loss,recon_loss=capsnet.calculate_loss(x_orig, x_hat, y_pred, y_true)
				val_loss=margin_loss.item()#print(loss)
				running_loss=running_loss+np.array([loss.item(),margin_loss,recon_loss.item()])
				Y['true'].extend(y_true.argmax(1).detach().cpu().numpy().tolist())
				Y['pred'].extend((y_pred**2).sum(2).argmax(1).detach().cpu().numpy().tolist())
			running_loss/=(i+1)
			Y['routing_weights'].iloc[:,:]=Y['routing_weights'].values/(i+1)

		Y['pred']=np.array(Y['pred']).astype(str)
		Y['true']=np.array(Y['true']).astype(str)
		#np.save('results/routing_weights/routing_weights.{}.npy'.format(epoch),Y['routing_weights'])
		pickle.dump(Y['routing_weights'],open('results/routing_weights/routing_weights.{}.p'.format(epoch),'wb'))
		Y['embeddings']=pd.DataFrame(PCA(n_components=2).fit_transform(np.vstack(Y['embeddings'])),columns=['x','y'])
		Y['embeddings2']=pd.DataFrame(PCA(n_components=2).fit_transform(np.vstack(Y['embeddings2'])),columns=['x','y'])
		#print(list(map(lambda x: x.shape,Y['embeddings3'])))
		Y['embeddings3']=pd.DataFrame(PCA(n_components=2).fit_transform(np.vstack(Y['embeddings3'])),columns=['x','y'])#'z'
		Y['embeddings']['color']=Y['true']
		Y['embeddings2']['color']=Y['true']
		Y['embeddings3']['color']=module_names*ma_v.beta.shape[0]#Y['true']
		Y['embeddings3']['name']=list(reduce(lambda x,y:x+y,[[i]*n_primary for i in Y['true']]))
		fig = px.scatter(Y['embeddings3'], x="x", y="y", color="color", symbol='name')#, text='name')
		py.plot(fig, filename='figures/embeddings3/embeddings3.{}.pos.html'.format(epoch),auto_open=False)
		#Y['embeddings3']['color']=list(reduce(lambda x,y:x+y,[[i]*n_primary for i in Y['true']]))
		fig = px.scatter(Y['embeddings3'], x="x", y="y", color="name")#, text='color')
		py.plot(fig, filename='figures/embeddings3/embeddings3.{}.true.html'.format(epoch),auto_open=False)
		fig = px.scatter(Y['embeddings'], x="x", y="y", color="color")
		py.plot(fig, filename='figures/embeddings/embeddings.{}.true.html'.format(epoch),auto_open=False)
		fig = px.scatter(Y['embeddings2'], x="x", y="y", color="color")
		py.plot(fig, filename='figures/embeddings2/embeddings2.{}.true.html'.format(epoch),auto_open=False)
		Y['embeddings'].loc[:,'color']=Y['pred']
		Y['embeddings2'].loc[:,'color']=Y['pred']
		fig = px.scatter(Y['embeddings'], x="x", y="y", color="color")
		py.plot(fig, filename='figures/embeddings/embeddings.{}.pred.html'.format(epoch),auto_open=False)
		fig = px.scatter(Y['embeddings2'], x="x", y="y", color="color")
		py.plot(fig, filename='figures/embeddings2/embeddings2.{}.pred.html'.format(epoch),auto_open=False)
		print('Epoch {}: Val Loss {}, Margin Loss {}, Recon Loss {}, Val R2: {}, Val MAE: {}'.format(epoch,running_loss[0],running_loss[1],running_loss[2],r2_score(Y['true'].astype(int),Y['pred'].astype(int)), mean_absolute_error(Y['true'].astype(int),Y['pred'].astype(int))))
		print(classification_report(Y['true'],Y['pred']))
		#Y=pd.DataFrame([])

if __name__=='__main__':
	main()
