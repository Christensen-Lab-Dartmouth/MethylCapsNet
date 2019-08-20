import torch
from torch import nn
from torch.nn import functional as F
from sklearn.preprocessing import LabelBinarizer
import numpy as np, pandas as pd

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

    @staticmethod
    def squash(x):
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

    @staticmethod
    def squash(x):
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

    def margin_loss(self,x, labels, weights=1.):
        batch_size = x.size(0)

        v_c = torch.sqrt((x**2).sum(dim=2, keepdim=True))

        #print(v_c)

        left = (F.relu(0.9 - v_c)**2).view(batch_size, -1)
        right = (F.relu(v_c - 0.1)**2).view(batch_size, -1)
        #print(left)
        #print(right)
        #print(labels)

        loss = weights*(labels * left + self.lr_balance * (1.0 - labels) * right)
        #print(loss.shape)
        loss = loss.sum(dim=1).mean()
        return loss

    def calculate_loss(self, x_orig, x_hat, y_pred, y_true, weights=1.):
        margin_loss = self.margin_loss(y_pred, y_true, weights=weights) # .expand_as(y_true)???
        recon_loss = self.gamma*self.recon_loss(x_orig,x_hat)
        loss = margin_loss + recon_loss
        return loss, margin_loss, recon_loss
