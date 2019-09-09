import torch
from torch import nn
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import classification_report
from torch.nn import functional as F
from sklearn.preprocessing import LabelBinarizer
import numpy as np, pandas as pd
import copy, os
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch import tensor
from sklearn.decomposition import PCA
import plotly.offline as py
import plotly.express as px
import pickle
import pysnooper
from torch.autograd import Variable, detect_anomaly
from functools import reduce
from torch.nn import BatchNorm1d
#from sksurv.linear_model.coxph import BreslowEstimator
from sklearn.utils.class_weight import compute_class_weight
from apex import amp
RANDOM_SEED=42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.multiprocessing.set_sharing_strategy('file_system')

class BCEWithNan(object):
	def __init__ (self, loss_function=nn.BCEWithLogitsLoss(reduce=False)):
		self.loss_function = loss_function
		self.replace_val=tensor([0.])
		if torch.cuda.is_available():
			self.replace_val=self.replace_val.cuda()

	def __call__ (self, y_hat, target, reduce=True):
		loss = self.loss_function(y_hat, target)
		d = torch.sum(~(torch.isnan(loss)), dim=1, keepdim=True).float()
		loss = torch.where(torch.isnan(loss), self.replace_val, loss)
		loss = torch.sum(loss, dim=1, keepdim=True) / d

		if reduce:
			return torch.mean(loss)
		else:
			return loss


class MethylationDataset(Dataset):
	def __init__(self, methyl_arr, outcome_col,binarizer=None, modules=[], module_names=None, original_interest_col=None):
		if binarizer==None:
			binarizer=LabelBinarizer()
			binarizer.fit(methyl_arr.pheno[outcome_col].astype(str).values)
		self.y=binarizer.transform(methyl_arr.pheno[outcome_col].astype(str).values)
		self.y_orig= np.argmax(self.y,1) # methyl_arr.pheno[outcome_col].values if np.issubdtype(methyl_arr.pheno[original_interest_col].dtype, np.number) else
		self.y_unique=np.unique(self.y_orig)#np.argmax(self.y,1))
		self.binarizer=binarizer
		if not modules:
			modules=[list(methyl_arr.beta)]
		self.modules=modules
		self.X=methyl_arr.beta
		print('Null val',self.X.isnull().sum().sum())
		self.length=methyl_arr.beta.shape[0]
		self.module_names=module_names
		self.pheno = methyl_arr.pheno

	def __len__(self):
		return self.length

	#@pysnooper.snoop('getitem.log')
	def __getitem__(self,i):
		X=[torch.FloatTensor(self.X.iloc[i].values)]
		modules=[torch.FloatTensor(self.X.iloc[i].loc[module].values) for module in self.modules] # .reshape(1,-1)
		y=[torch.FloatTensor(self.y[i])]
		#y_orig=[torch.FloatTensor(self.y_orig[i].reshape(1,1))]
		return tuple(X+modules+y)#+y_orig)

def softmax(input_tensor, dim=1):
	transposed_input = input_tensor.transpose(dim, len(input_tensor.size()) - 1)
	softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
	return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input_tensor.size()) - 1)

class CoxLoss(nn.Module):
	""""""# add custom loss https://github.com/Tombaugh/CapSurv/blob/master/capsurv.py
	def __init__(self):
		super(CoxLoss, self).__init__()

	def forward(self, y_pred_caps, y_true, y_true_orig):
		y_pred_caps=y_pred_caps[:,1,:]
		hazard_ratio=torch.exp(y_pred_caps)#torch.norm(y_pred_caps, p=2, dim=1))
		log_risk=torch.log(torch.cumsum(hazard_ratio,dim=1))
		uncensored_likelihood = y_pred_caps - log_risk
		loss = uncensored_likelihood * -1.
		return loss.mean(0)

class MLP(nn.Module): # add latent space extraction, and spits out csv line of SQL as text for UMAP
	def __init__(self, n_input, hidden_topology, dropout_p=0., n_outputs=1, binary=False, softmax=False, relu=True):
		super(MLP,self).__init__()
		self.hidden_topology=hidden_topology
		self.topology = [n_input]+hidden_topology+[n_outputs]
		layers = [nn.Linear(self.topology[i],self.topology[i+1]) for i in range(len(self.topology)-2)]
		for layer in layers:
			torch.nn.init.xavier_uniform_(layer.weight)
		self.layers = [nn.Sequential(layer,nn.ReLU(),nn.BatchNorm1d(layer.out_features)) for layer in layers]
		self.output_layer = nn.Linear(self.topology[-2],self.topology[-1])
		torch.nn.init.xavier_uniform_(self.output_layer.weight)
		if binary:
			output_transform = nn.Sigmoid()
		elif softmax:
			output_transform = nn.Softmax()
		elif relu:
			output_transform = nn.ReLU()
		else:
			output_transform = nn.Dropout(p=0.)
		self.layers.append(nn.Sequential(self.output_layer,output_transform))
		self.mlp = nn.Sequential(*self.layers)

	def forward(self, x):
		#print(x.shape)
		return self.mlp(x)

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

	#@staticmethod
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
		self.u_hat=copy.deepcopy(u_hat.squeeze(4).detach().cpu())
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

	def return_embedding_previous_layer(self):
		return self.u_hat.numpy()[...,0]

	#@staticmethod
	def squash(self,x):
		#print(x.size())
		squared_norm = (x ** 2).sum(-1, keepdim=True)
		#print('norm',squared_norm.size())
		output_tensor = squared_norm *  x / ((1. + squared_norm) * torch.sqrt(squared_norm))
		return output_tensor

class Decoder(nn.Module):
	def __init__(self, n_input, n_output, hidden_topology):
		super(Decoder, self).__init__()
		self.decoder=MLP(n_input,hidden_topology, 0., n_outputs=n_output, binary=False, relu=False)

	def forward(self, x):
		return self.decoder(x)

class CapsNet(nn.Module):
	def __init__(self, primary_caps, caps_hidden_layers, caps_output_layer, decoder, lr_balance=0.5, gamma=0.005):
		super(CapsNet, self).__init__()
		self.primary_caps=primary_caps
		self.caps_hidden_layers=caps_hidden_layers
		self.caps_output_layer=caps_output_layer
		self.decoder=decoder
		self.recon_loss_fn = BCEWithNan()#nn.BCEWithLogitsLoss() # WithLogits https://github.com/shllln/BCEWithNan
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

	def margin_loss(self,x, labels, weights=1.):
		batch_size = x.size(0)

		v_c = torch.sqrt((x**2).sum(dim=2, keepdim=True))

		#print(v_c)

		left = (F.relu(0.9 - v_c)**2).view(batch_size, -1)
		right = (F.relu(v_c - 0.1)**2).view(batch_size, -1)
		#print(left)
		#print(right)
		#print(labels)

		loss = labels * left + self.lr_balance * (1.0 - labels) * right#weights*(labels * left + self.lr_balance * (1.0 - labels) * right)
		#print(loss.shape)
		loss = loss.sum(dim=1).mean()
		return loss

	#@pysnooper.snoop('loss.log')
	def calculate_loss(self, x_orig, x_hat, y_pred, y_true, weights=1.):
		margin_loss = self.margin_loss(y_pred, y_true, weights=weights) # .expand_as(y_true)???
		recon_loss = self.gamma*self.recon_loss(x_orig.squeeze(1),x_hat)
		loss = margin_loss + recon_loss
		return loss, margin_loss, recon_loss

class Trainer:
	def __init__(self, capsnet, validation_dataloader, n_epochs, lr, n_primary, custom_loss, gamma2, class_balance=False):
		self.capsnet=capsnet
		self.validation_dataloader = validation_dataloader
		self.lr = lr
		self.optimizer = Adam(self.capsnet.parameters(),self.lr)
		self.capsnet, self.optimizer = amp.initialize(self.capsnet, self.optimizer, opt_level='O0', loss_scale=1.0)#'dynamic'
		self.scheduler=CosineAnnealingLR(self.optimizer, T_max=10, eta_min=0, last_epoch=-1)
		self.n_epochs = n_epochs
		self.module_names = self.validation_dataloader.dataset.module_names
		self.n_primary=n_primary
		self.custom_loss=custom_loss
		self.gamma2=gamma2
		self.custom_loss_fn=dict(none=None,
								  cox=CoxLoss())[self.custom_loss]
		self.class_balance = class_balance

	def compute_custom_loss(self,y_pred_caps, y_true, y_true_orig):
		if self.custom_loss=='none':
			return 0.
		else:
			loss = self.custom_loss_fn(y_pred_caps, y_true, y_true_orig)
			return loss

	def initialize_dirs(self):
		for d in ['figures/{}'.format(x) for x in ['embedding_primarycaps_aligned','embedding_primarycaps','embedding_primarycaps_cat','embedding_outputcaps']]:
			os.makedirs(d,exist_ok=True)
		os.makedirs('results/routing_weights',exist_ok=True)

	#@pysnooper.snoop('fit_model.log')
	def fit(self, dataloader):
		self.initialize_dirs()
		if self.class_balance:
			self.weights=torch.tensor(compute_class_weight('balanced',np.arange(len(dataloader.dataset.binarizer.classes_)),np.argmax(dataloader.dataset.y,axis=1)),dtype=torch.float)
			if torch.cuda.is_available():
				self.weights = self.weights.cuda()
		else:
			self.weights=1.
		self.losses=dict(train=[],val=[])
		best_model = self.capsnet
		self.val_losses=[]
		for epoch in range(self.n_epochs):
			self.epoch=epoch
			self.losses['train'].append(self.train_loop(dataloader))
			val_loss=self.val_test_loop(self.validation_dataloader)[0]
			self.val_losses.append(val_loss[0])
			self.losses['val'].append(val_loss)
			if val_loss[0]<=min(self.val_losses):
				best_model=copy.deepcopy(self.capsnet)
		self.capsnet=best_model
		return self

	def predict(self, dataloader):
		self.initialize_dirs()
		self.epoch='Test'
		test_loss,Y=self.val_test_loop(dataloader)
		return Y

	#@pysnooper.snoop('train_loop.log')
	def train_loop(self, dataloader):
		self.capsnet.train(True)
		running_loss=0.
		Y={'true':[],'pred':[]}
		n_batch=(len(dataloader.dataset.y_orig)//dataloader.batch_size)
		for i,batch in enumerate(dataloader):
			x_orig=batch[0]
			#print(x_orig)
			y_true=batch[-1]#[-2]
			#y_true_orig=batch[-1]
			module_x = batch[1:-1]#2]
			if torch.cuda.is_available():
				x_orig=x_orig.cuda()
				y_true=y_true.cuda()
				#y_true_orig=y_true_orig.cuda()
				module_x=[mod.cuda() for mod in module_x]
			x_orig, x_hat, y_pred, embedding, primary_caps_out=self.capsnet(x_orig,module_x)
			loss,margin_loss,recon_loss=self.capsnet.calculate_loss(x_orig, x_hat, y_pred, y_true, weights=self.weights)
			#loss=loss+self.gamma2*self.compute_custom_loss(y_pred, y_true, y_true_orig)
			self.optimizer.zero_grad()
			with amp.scale_loss(loss,self.optimizer) as scaled_loss, detect_anomaly():
				scaled_loss.backward()
			#loss.backward()
			self.optimizer.step()
			Y['true'].extend(y_true.argmax(1).detach().cpu().numpy().flatten().astype(int).tolist())
			Y['pred'].extend(F.softmax(torch.sqrt((y_pred**2).sum(2))).argmax(1).detach().cpu().numpy().astype(int).flatten().tolist())
			train_loss=margin_loss.item()#print(loss)
			print('Epoch {} [{}/{}]: Train Loss {}'.format(self.epoch,i,n_batch,train_loss))
			running_loss+=train_loss

		#y_true,y_pred=Y['true'],Y['pred']
		running_loss/=(i+1)
		print('Epoch {}: Train Loss {}, Train R2: {}, Train MAE: {}'.format(self.epoch,running_loss,r2_score(Y['true'],Y['pred']), mean_absolute_error(Y['true'],Y['pred'])))
		print(classification_report(Y['true'],Y['pred']))
		#print(capsnet.primary_caps.get_weights())
		self.scheduler.step()
		return running_loss

	#@pysnooper.snoop('val_loop.log')
	def val_test_loop(self, dataloader):
		self.capsnet.train(False)
		running_loss=np.zeros((3,)).astype(float)
		n_batch=int(np.ceil(len(dataloader.dataset.y_orig)/dataloader.batch_size))
		Y={'true':[],'pred':[],'embedding_primarycaps_aligned':[],'embedding_primarycaps':[],'embedding_primarycaps_cat':[],'embedding_outputcaps':[],'routing_weights':[]}
		with torch.no_grad():
			for i,batch in enumerate(dataloader):
				x_orig=batch[0]
				y_true=batch[-1]#2
				#y_true_orig=batch[-1]
				module_x = batch[1:-1]#2
				if torch.cuda.is_available():
					x_orig=x_orig.cuda()
					y_true=y_true.cuda()
					#y_true_orig=y_true_orig.cuda()
					module_x=[mod.cuda() for mod in module_x]
				x_orig, x_hat, y_pred, embedding, primary_caps_out=self.capsnet(x_orig,module_x)
				loss,margin_loss,recon_loss=self.capsnet.calculate_loss(x_orig, x_hat, y_pred, y_true, weights=self.weights)
				#loss=loss+self.gamma2*self.compute_custom_loss(y_pred, y_true, y_true_orig)
				val_loss=margin_loss.item()#print(loss)
				print('Epoch {} [{}/{}]: Val Loss {}'.format(self.epoch,i,n_batch,val_loss))
				running_loss=running_loss+np.array([loss.item(),margin_loss,recon_loss.item()])

				routing_coefs=self.capsnet.caps_output_layer.return_routing_coef().detach().cpu().numpy()
				if not i:
					Y['routing_weights']=pd.DataFrame(routing_coefs[0,...,0].T,index=dataloader.dataset.binarizer.classes_,columns=dataloader.dataset.module_names)
				else:
					Y['routing_weights']+=pd.DataFrame(routing_coefs[0,...,0].T,index=dataloader.dataset.binarizer.classes_,columns=dataloader.dataset.module_names)
				Y['embedding_primarycaps'].append(torch.cat([primary_caps_out[i] for i in range(x_orig.size(0))],dim=0).detach().cpu().numpy())
				primary_caps_out=primary_caps_out.view(primary_caps_out.size(0),primary_caps_out.size(1)*primary_caps_out.size(2))
				Y['embedding_outputcaps'].append(embedding.detach().cpu().numpy())
				Y['embedding_primarycaps_cat'].append(primary_caps_out.detach().cpu().numpy())
				Y['embedding_primarycaps_aligned'].append(self.capsnet.caps_output_layer.return_embedding_previous_layer())
				Y['true'].extend(y_true.argmax(1).detach().cpu().numpy().astype(int).flatten().tolist())
				Y['pred'].extend((y_pred**2).sum(2).argmax(1).detach().cpu().numpy().astype(int).flatten().tolist())
			running_loss/=(i+1)
			Y['routing_weights'].iloc[:,:]=Y['routing_weights'].values/(i+1)
			Y['pred']=np.array(Y['pred']).astype(str)
			Y['true']=np.array(Y['true']).astype(str)
			print('Epoch {}: Val Loss {}, Margin Loss {}, Recon Loss {}, Val R2: {}, Val MAE: {}'.format(self.epoch,running_loss[0],running_loss[1],running_loss[2],r2_score(Y['true'].astype(float),Y['pred'].astype(float)), mean_absolute_error(Y['true'].astype(float),Y['pred'].astype(float))))
			print(classification_report(Y['true'],Y['pred']))
			self.make_plots(Y, dataloader)
			self.save_routing_weights(Y)
		return running_loss, Y

	def make_plots(self, Y, dataloader):
		for k in ['embedding_primarycaps','embedding_primarycaps_aligned']:
			Y[k]=pd.DataFrame(PCA(n_components=2).fit_transform(np.vstack(Y[k])),columns=['x','y'])
			Y[k]['pos']=self.module_names*dataloader.dataset.y.shape[0]#ma_v.beta.shape[0]#Y['true']
			Y[k]['true']=list(reduce(lambda x,y:x+y,[[self.module_names[i]]*self.n_primary for i in Y['true']]))
			for k2 in ['pos','true']:
				fig = px.scatter(Y[k], x="x", y="y", color=k2)#, text='color')
				py.plot(fig, filename='figures/{0}/{0}.{1}.{2}.html'.format(k,self.epoch,k2),auto_open=False)

		for k in ['embedding_outputcaps','embedding_primarycaps_cat']:
			Y[k]=pd.DataFrame(PCA(n_components=2).fit_transform(np.vstack(Y[k])),columns=['x','y'])
			for k2 in ['true','pred']:
				Y[k]['color']=Y[k2]
				fig = px.scatter(Y[k], x="x", y="y", color="color")
				py.plot(fig, filename='figures/{0}/{0}.{1}.{2}.html'.format(k,self.epoch,k2),auto_open=False)

	def save_routing_weights(self, Y):
		pickle.dump(Y['routing_weights'],open('results/routing_weights/routing_weights.{}.p'.format(self.epoch),'wb'))
