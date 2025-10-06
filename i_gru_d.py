import torch
import math
import numpy as np


class GRUD_cell(torch.nn.Module):
	def __init__(self,
			input_size,	#take an integer, the number of input features	
			hidden_size,	#take an integer, the number of neurons in hidden layer
			output_size,	#take an integer, the number of output features
			x_mean=0,	#x_mean should be a vector (rather than a scalar as the default value here) containing the empirical mean value for each feature
			update_x_mean=True,
			non_neg=None,
			#bias=True,	#always include bias vector for z,r,h
			#bidirectional=False,	#not applicable in this study since in practice we won't know future data points
			#dropout_type='mloss',
			dropoutratio=0,	#the dropout ratio for between timesteps of hidden layers.
			dtypearg=torch.float64,
			usedevice='cuda:0'
			):
		torch.set_default_dtype(dtypearg)
		cuda_available = torch.cuda.is_available()	#return True if NVIDIA available
		device = torch.device(usedevice if cuda_available else 'cpu')
		self.dtype=dtypearg
		self.device=device
		super(GRUD_cell,self).__init__()
		self.input_size=input_size
		self.hidden_size=hidden_size
		self.output_size=output_size
		#as noted by the author, register_buffer is used because it automatically load data to GPU,if any. However, this also prohibit x_mean from updating
		if update_x_mean:
			self.x_mean = torch.nn.Parameter(torch.tensor(x_mean.clone(), dtype=dtypearg, requires_grad=True, device=device))
		else:
			self.register_buffer('x_mean', torch.tensor(x_mean.clone(), dtype=dtypearg, device=device))
		#self.bias=bias
		self.dropoutratio=dropoutratio
		self.dropoutlayer=torch.nn.Dropout(p=dropoutratio)	#dropout must be defined within __init__ so switch between eval() and train() model effectively turn off/on dropout			
		#self.dropout_type=dropout_type
		#self.bidirectional=bidirectional
		#num_directions=2 if bidirectional else 1
		#weight matrix for gamma
		self.w_dg_x=torch.nn.Linear(input_size,input_size,bias=True,device=device)
		self.w_dg_h=torch.nn.Linear(input_size,hidden_size,bias=True,device=device)
		#weight matrix for z, update gate
		self.w_xz=torch.nn.Linear(input_size,hidden_size,bias=False,device=device)
		self.w_hz=torch.nn.Linear(hidden_size,hidden_size,bias=False,device=device)
		self.w_mz=torch.nn.Linear(input_size,hidden_size,bias=False,device=device)
		self.b_z=torch.nn.Parameter(torch.tensor(np.ndarray(hidden_size),dtype=dtypearg,requires_grad=True,device=device))
		#weight matrix for r, reset gate
		self.w_xr=torch.nn.Linear(input_size,hidden_size,bias=False,device=device)
		self.w_hr=torch.nn.Linear(hidden_size,hidden_size,bias=False,device=device)
		self.w_mr=torch.nn.Linear(input_size,hidden_size,bias=False,device=device)
		self.b_r=torch.nn.Parameter(torch.tensor(np.ndarray(hidden_size),dtype=dtypearg,requires_grad=True,device=device))
		#weight matrix for hidden units
		self.w_xh=torch.nn.Linear(input_size,hidden_size,bias=False,device=device)
		self.w_hh=torch.nn.Linear(hidden_size,hidden_size,bias=False,device=device)
		self.w_mh=torch.nn.Linear(input_size,hidden_size,bias=False,device=device)
		self.b_h=torch.nn.Parameter(torch.tensor(np.ndarray(hidden_size),dtype=dtypearg,requires_grad=True,device=device))
		#weight matrix for linking hidden layer to final output
		self.w_hy=torch.nn.Linear(hidden_size,output_size,bias=True,device=device)
		self.batchnorm=torch.nn.BatchNorm1d(hidden_size)
		#the hidden state vector, it is the conveying belt that transfer information to the next timestep
		#it will be scaled to batch_size x hidden_size during forward implementation (as the batch_size is unknown here)
		Hidden_State=torch.zeros(hidden_size,dtype=dtypearg,device=device)
		#it is registered in buffer, which prevents it from change. so this is a "stateless" model, which assumes no relatinoship between each training sample
		self.register_buffer('Hidden_State',Hidden_State)
		#a vector containing the last observation
		#it will be scaled to [batch_size, input_size] in the following script
		X_last_obs=torch.zeros(input_size,dtype=dtypearg,device=device)
		self.register_buffer('X_last_obs',X_last_obs)
		self.non_neg=non_neg
		self.reset_parameters()
	def reset_parameters(self):	#this reset all weight parameters	
		stdv=1.0/math.sqrt(self.hidden_size)
		#for weight in self.parameters():	#note this reset all weights matrix except those registered in buffer
			#torch.nn.init.uniform_(weight, -1 * stdv, stdv)
		for name,weight in self.named_parameters():
			if(name != 'x_mean'):	#avoid reset certain weights at the startup
				torch.nn.init.uniform_(weight, -1 * stdv, stdv)
	def setdevice(self,usedevice):
		self.to(usedevice)
		self.device=torch.device(usedevice)
	@property
	def _flat_weights(self):	#no idea what this is doing
		return list(self._parameters.values())
	def forward(self, input):
		#determine the device where 
		device=self.device	#<class 'torch.device'>
		#input has these dimensions [batch_size,"X Mask Delta",feature_size, timestep_size]
		#move input to corresponding device
		if(input.device != device):
			input=input.to(device)
		X=input[:,0,:,:]	#X has dimension [batch_size, feature_size, timestep_size]
		Mask=input[:,1,:,:]
		Delta=input[:,2,:,:]
		step_size=X.size(2)	#the size of timesteps
		output=None
		h=getattr(self,'Hidden_State')	#h is a vector
		x_mean=getattr(self,'x_mean')
		x_last_obsv=getattr(self,'X_last_obs')
		#an empty tensor for holding the output of each timestep
		#the dimensions are [batch_size,timestep_size,output_feature_size]
		output_tensor=torch.empty([X.size(0),X.size(2),self.output_size], 
				dtype=X.dtype, device=device)
		#an empty tensor for holding the hidden state of each timestep
		#the dimensions are [batch_size,timestep_size,hidden_size]
		hidden_tensor=torch.zeros([X.size(0),X.size(2),self.hidden_size],
				dtype=X.dtype,device = device)
		#x_tensor holds the original x values, and if not available, those imputed on the fly
		#[batch_size, feature_size, timestep_size]
		x_tensor=torch.zeros([X.size(0),self.input_size,X.size(2)],
				dtype=X.dtype,device = device)
		#iterate over timesteps
		for timestep in range(X.size(2)):
			#squeeze drop the timestep dimension and ends up with [batch_size,feature_size]
			x=X[:,:,timestep]	#x has dimension [batch_size, feature_size]
			m=Mask[:,:,timestep]
			d=Delta[:,:,timestep]
			#the gamma vector for filtering x. the dimension is [batch_size,feature_size]
			#each element in gamma_x is a value within (0,1]
			gamma_x=torch.exp(-1*torch.nn.functional.relu(self.w_dg_x(d)))
			#the gamma vector for filtering h. the dimension is [batch_size,hidden_size]
			gamma_h=torch.exp(-1*torch.nn.functional.relu(self.w_dg_h(d)))
			#update x_last_obsv. x_last_obsv is a vector of input_size
			#it is changed to a [batch_size,feature_size] tensor on first run
			#x_last_obsv are all 0 at the beginning, and update to the latest available data 
			x_last_obsv=torch.where(m>0,x,x_last_obsv)
			#set nan in x to 0 -- necessary to do this on the training data before input to cell
			#x[torch.isnan(x)]=0	#this kind of operation is strictly not allowed becuase it raises the RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
			x=m*x + (1-m)*(gamma_x*x_last_obsv + (1-gamma_x)*x_mean)
			if(not self.non_neg is None):
				x[:,self.non_neg]=torch.nn.functional.softplus(x[:,self.non_neg],beta=100)	#when beta is 100, x=0 map to 0.007
			#
			h=gamma_h*h	#h is initialized as a vector, and reshaped to [batch_size,hidden_size] on first run
			#z is [batch_size, hidden_size]
			z=torch.sigmoid(self.w_xz(x) + self.w_hz(h) + self.w_mz(m) + self.b_z)
			#r is [batch_size, hidden_size]
			r=torch.sigmoid(self.w_xr(x) + self.w_hr(h) + self.w_mr(m) + self.b_r)
			#h_tilde is [batch_size, hidden_size]
			h_tilde=torch.tanh(self.w_xh(x) + self.w_hh(r*h) + self.w_mh(m) + self.b_h)
			#h is [batch_size, hidden_size]
			h=(1-z)*h + z*h_tilde
			#if using batch normalization, should be added here
			h=self.batchnorm(h)
			if(self.dropoutratio > 0):	#remember to use eval() to turn off dropout during evaluation
				h=self.dropoutlayer(h)
			step_output=self.w_hy(h)		#[batch_size,output_size]
			step_output=torch.sigmoid(step_output)
			#note the following operations generate hard copy of the right operand, which is important
			output_tensor[:,timestep,:]=step_output	#[batch_size,timestep_size,output_size]
			hidden_tensor[:,timestep,:]=h	#[batch_size,timestep_size,hidden_size]
			x_tensor[:,:,timestep]=x	#[batch_size, feature_size, timestep_size]
		#end of for
		output=(output_tensor,hidden_tensor,x_tensor,Mask)
		return output
	#end of forward
#end of class





class GRUD_model(torch.nn.Module):
	def __init__(self,
			input_size,
			hidden_size,	#this determines hidden_size of both the first and all stacked layers
			output_size,	#the output size of every layer
			num_layers=1,
			x_mean=0,	#empirical mean of each input feature
			update_x_mean=True,
			non_neg=None,
			#bias=True,
			#batch_first=False,
			#bidirectional=False,
			#dropout_type='mloss',
			dropoutratio=0,
			dtypearg=torch.float64,
			usedevice='cuda:0'
			):
		torch.set_default_dtype(dtypearg)
		cuda_available = torch.cuda.is_available()
		device = torch.device(usedevice if cuda_available else 'cpu')
		self.device=device
		super(GRUD_model,self).__init__()
		#first layer is the GRU-D
		self.gru_d=GRUD_cell(input_size=input_size,
					hidden_size=hidden_size,
					output_size=output_size,
					x_mean=x_mean,
					update_x_mean=update_x_mean,
					non_neg=non_neg,
					dropoutratio=dropoutratio,
					dtypearg=dtypearg,
					usedevice=usedevice
					)
		self.num_layers=num_layers
		self.hidden_size=hidden_size
		#stack other layers as regular GRU layer
		if(self.num_layers > 1):
			self.gru_layers=torch.nn.GRU(input_size=hidden_size,
				hidden_size=hidden_size,
				batch_first=True,	#necessary because the output from gru_d is [batch_size,timestep,feature], whereas torch.nn.GRU takes [timestep,batch,feature] by default
				num_layers=self.num_layers-1,
				dropout=dropoutratio,
				device=device
				)	#I manually confirmed torch.nn.GRU dropout is sensitive to switch between eval() and training()
			#this is for converting the last layer hidden output to actual output
			self.hidden_to_output=torch.nn.Linear(hidden_size, output_size, bias=True)	
	#end of __init__ 
	def setdevice(self,usedevice):
		self.to(usedevice)
		self.device=torch.device(usedevice)
		self.gru_d.setdevice(usedevice)
	def _flat_weights(self):
		return list(self._parameters.values())
	def forward(self,input):
		#pass through the first gru_d layer
		#determine the device where 
		if(hasattr(self,'device')):
			device=self.device	#<class 'torch.device'>
			#move input to corresponding device
			if(input.device != device):
				input=input.to(device)
		(real_output,hidden,x_tensor,m_tensor)=self.gru_d(input)
		#real_output is [batch_size,timestep,output_feature]
		#hidden is [batch_size,timestep,hidden_size]
		#x_tensor [batch_size, feature_size, timestep_size]
		if(self.num_layers > 1):
			#pass through the rest of regular gru layers
			#output contains the final layer output for all samples and all timesteps
			#output has shape [batch_size,timestep,hidden_size]
			#hidden has shape [num_layers, batch_size, hidden_size]
			#in this example output[:,timestep-1,:] == hidden, should be all true
			(output,hidden)=self.gru_layers(hidden)
			#covert output of last layer hidden output [batch_size, timestep, hidden_size] to real output [batch_size, timestep, output_feature]
			real_output=self.hidden_to_output(output)
			real_output=torch.sigmoid(real_output)
			hidden=output	#hidden now is [batch_size, timestep, hidden_size]
		#end of if
		return(real_output,hidden,x_tensor,m_tensor)	#make sure the output shape is same as gru_d
	#end of forward
#end of class

class I_GRUD_model(torch.nn.Module):
	def __init__(self,
			input_size,	#the input feature size
			hidden_size,	#this determines hidden_size of both the first and all stacked layers
			output_size,	#the output size of every layer
			num_layers=1,
			x_mean=0,	#empirical mean of each input feature
			update_x_mean=True,
			#bias=True,
			#bidirectional=False,
			dropoutratio=0,
			betadropoutratio=0,
			traceback=200,
			non_neg=None,
			dtypearg=torch.float64,
			interpretable_mode=True, #whether turn on interpretable mode
			usedevice='cuda:0'
		):
		torch.set_default_dtype(dtypearg)
		cuda_available = torch.cuda.is_available()	#return True if NVIDIA available
		device = torch.device(usedevice if cuda_available else 'cpu')
		self.dtype=dtypearg
		self.device=device
		super(I_GRUD_model,self).__init__()
		self.input_size=input_size
		self.hidden_size=hidden_size
		self.output_size=output_size
		self.traceback=traceback
		self.interpretable_mode=interpretable_mode
		self.batchnorm=torch.nn.BatchNorm1d(input_size)
		self.grud_model=GRUD_model(input_size,hidden_size,output_size,
					num_layers=num_layers,x_mean=x_mean,
					update_x_mean=update_x_mean,non_neg=non_neg,
					dropoutratio=dropoutratio,dtypearg=dtypearg,
					usedevice=usedevice)
		if(self.interpretable_mode):
			#this following codes are for attention 
			self.w_beta=torch.nn.Linear(hidden_size,input_size,bias=False,device=device)	#feature wise attention
			self.w_epsilon=torch.nn.Linear(hidden_size, 1, bias=False, device=device)	#timestep wise attention
			self.softmax=torch.nn.Softmax(dim=1)
			#weight matrix for linking context vector (attention version only) to final output
			self.w_cy=torch.nn.Linear(input_size,output_size,bias=True,device=device)
			self.betadropoutlayer=torch.nn.Dropout(p=betadropoutratio)
		#end of if
		self.reset_parameters()
	def reset_parameters(self):	#this reset all weight parameters	
		stdv=1.0/math.sqrt(self.hidden_size)
		#for weight in self.parameters():	#note this reset all weights matrix except those registered in buffer
			#torch.nn.init.uniform_(weight, -1 * stdv, stdv)
		for name,weight in self.named_parameters():
			if(name != 'x_mean'):	#avoid reset certain weights at the startup
				torch.nn.init.uniform_(weight, -1 * stdv, stdv)
	def setdevice(self,usedevice):
		self.to(usedevice)
		self.device=torch.device(usedevice)
		self.grud_model.setdevice(usedevice)
	@property
	def _flat_weights(self):	#no idea what this is doing
		return list(self._parameters.values())
	def forward(self, input):
		#determine the device where 
		device=self.device	#<class 'torch.device'>
		#move input to corresponding device
		if(input.device != device):
			input=input.to(device)
		(real_output,hidden,x_tensor,m_tensor)=self.grud_model(input)
		#real_output [batch_size,timestep,output_size]	
		#hidden [batch_size, timestep, hidden_size]
		if(not self.interpretable_mode):
			return (real_output,hidden,x_tensor,m_tensor)
		else:
			batch_size=real_output.size(0)
			timestep_size=real_output.size(1)
			#an empty tensor for holding the output of each timestep
			#the dimensions are [batch_size,timestep_size,output_feature_size]
			output_tensor=torch.empty([batch_size,timestep_size,self.output_size], 
					dtype=input.dtype, device=device)
			context_tensor=torch.empty([batch_size,timestep_size,self.output_size], 
					dtype=input.dtype, device=device)
			#visit level attention has two parts
			#epsilon_tensor is for holding original output [batch_size,timestep_size]
			#alpha_tensor is softmax activation of epsilon, which provides timestep level attention
			epsilon_tensor=torch.zeros([batch_size,timestep_size],
					dtype=input.dtype,device = device) 
			#feature level attention [batch,input,timestep]
			beta_tensor=torch.zeros([batch_size,self.input_size,timestep_size],
					dtype=input.dtype,device = device) 
			#iterate over timesteps
			for timestep in range(timestep_size):
				h=hidden[:,timestep,:]	#[batch, hidden]
				#********************************************#
				#the following codes are RETAIN-like attention mechanism
				#visit level attention
				epsilon=self.w_epsilon(h)	#[batch,1]
				epsilon_tensor[:,timestep]=epsilon[:,0]	#store epsilon for each timestep
				#softmax epsilon to get alpha attention
				attentionsta=timestep - self.traceback	#consider limited timesteps back
				if(attentionsta < 0):attentionsta=0
				alpha=self.softmax(epsilon_tensor[:,attentionsta:(timestep+1)].clone()) #[batch,ts]
				#convert [batch,ts] to [batch,input,ts] for easy muliplication with beta and x
				alpha=alpha[:,None,:]	#convert to [batch,1,ts] note! reference only
				alpha=alpha.expand(alpha.shape[0],self.input_size,alpha.shape[2])	#expand to [batch,input,ts] note ! reference only, so now every input feature in a timestep got same alpha
				#feature level attention
				beta=self.w_beta(h)	#[batch,input_size]
				beta=5 * torch.tanh(beta/5)	#smooth out beta as I found beta maybe super large in certain training fold
				beta=self.betadropoutlayer(beta)
				#beta=self.batchnorm(beta)
				beta_tensor[:,:,timestep]=beta	#store beta [batch,input,ts]
				#multiply alpha, beta, and x for all current timesteps
				#!!! clone is necessary
				abx=torch.mul(alpha,torch.mul(beta_tensor[:,:,attentionsta:(timestep + 1)].clone(),x_tensor[:,:,attentionsta:(timestep + 1)].clone())) #[batch,input,ts]
				#context is weighted (already done) average (or sum) of traceback timesteps
				context=torch.sum(abx,dim=2)	#average existing timesteps values for each input feature [batch,input]
				#step_output=torch.sigmoid(self.w_cy(context)) #[batch,output]
				context_sum=torch.sum(context,dim=1)	#[batch]
				context_tensor[:,timestep,:]=context_sum[:,None]	#[batch,timestep,output]
				step_output=torch.sigmoid(context_sum) #[batch]
				step_output=step_output[:,None]		#[batch,output]
				output_tensor[:,timestep,:]=step_output	#[batch,timestep,output]
			#end of for
			output=(output_tensor,hidden,epsilon_tensor,beta_tensor,x_tensor,context_tensor,m_tensor)
			return output
		#end of if
	#end of forward
#end of class



from torch.utils.data import Dataset
from torch.utils.data import DataLoader
class datasetgenerator(Dataset):
	def __init__(self,nparray,targetarray,valid_timesteps_array=None):
		assert(nparray.shape[0] == targetarray.shape[0]), "nparray length and target array length not match"
		self.z=None
		if(not valid_timesteps_array is None):
			assert(nparray.shape[0] == len(valid_timesteps_array)), "nparray length and valid timestep array length not match"
			self.z=valid_timesteps_array
		self.x=nparray
		self.y=targetarray
	def __len__(self):
		return(self.x.shape[0])
	def __getitem__(self,idx):
		if(self.z is None):
			return(self.x[idx,:,:,:],self.y[idx])
		else:
			return(self.x[idx,:,:,:],self.y[idx],self.z[idx])
		#end of if
	#end of def
#end of class



