import numpy as np

#Convolutional Neural Network
class CNN:
{

	"""
	Architecture :
	Conv1 -> Conv2 -> Pool -> FC1 -> FC2
	ReLU after every Conv layer and FC layer

	Conv1 -> F = 3((3*3*1)*32 filters), S=1, P = 0
			 I/P -> 48*48  ;  O/P -> 46*46*32

	Conv2  -> F = 3((3*3*32)*64 filters, S=1, P=0
			 I/P -> 46*46*32 ;	O/P  ->	44*44*64

	Pool  -> Max Pooling (2*2)
			 I/P -> 44*44*64 ; 	O/P -> 22*22*64

	FC1   -> (22*22*64)*256		
			O/P -> 1*1*256

	FC2   -> (256*7)
			O/P  -> 1*1*7	-> class probabilities

	Softmax Loss, L2 Regularization loss
	"""
	W=np.empty(5,dtype='int')
	H=np.empty(5,dtype='int')
	D=np.empty(5,dtype='int')
	SIZE=np.empty(5,dtype='int')
	F=np.empty(5,dtype='int')
	S=np.empty(5,dtype='int')
	P=np.empty(5,dtype='int')

	WT=np.empty(5,dtype=object)
	B=np.empty(5,dtype=object)

	#Width, height and depth of image
	W[0] = 48
	H[0] = 48
	D[0] = 1
	SIZE[0] = 1

	#Convolution layer - 1 parameters
	#Fi - Filter size, Si - Stride, Pi - Padding, WTi, Bi - Weight and Bias for the layer
	#SIZEi - No. of filters/ depth of stacked layers
	F[1] = 3
	S[1] = 1
	P[1] = 0
	SIZE[1] = 32

	#Convolution layer - 2 parameters	
	F[2] = 3
	S[2] = 1
	P[2] = 0
	SIZE[2] = 64

	#Max Pooling (2*2 pooling)
	F[3] = 2
	S[3] = 1
	SIZE[3] = SIZE[2]

	#Fully Connected Layer - 1
	SIZE[4] = 256

	#Fully Connected Layer - 2
	#no. of categories
	SIZE[5] = 7

	#CNN parameters and initialization
	# n = no. of training examples
	def __init__(self,n):

		self.num_example = n
		self.calculate_parameters()

		#Randomly initialize from uniform distribution(randn)
		#Mean = 0 and variance = 2/num_example  for better convergence with ReLU neurons
		self.WT[1] = np.random.randn(F[1],F[1],D[0],SIZE[1])*sqrt(2.0/num_example)
		self.B[1] = np.zeros(1,1,1,SIZE[1])

		self.WT[2] = np.random.randn(F[2],F[2],D[1],SIZE[2])*sqrt(2.0/num_example)
		self.B[2] = np.zeros(1,1,1,SIZE[2])

		#No hyperparameter for pooling layer 

		self.WT[4] = 0.01*np.random.randn(W[3],H[3],D[3],SIZE[4])
		self.B[4] = np.zeros(1,1,1,SIZE[4])

		self.WT[5] = 0.01*np.random.randn(W[4],H[4],D[4],SIZE[5])
		self.B[5] = np.zeros(1,1,1,SIZE[5])


	def calculate_parameters(self):

		#Output size of 1st layer
		#I/P - (W0*H0*SIZE0) ; O/P - W1*H1*SIZE1 ; WT1 = (F1*F1*D0)*SIZE1
		self.W[1] = (self.W[0]-self.F[1]+2*self.P[1])/self.S[1] + 1
		self.H[1] = self.W[1]
		self.D[1] = self.SIZE[1]

		#I/P - (W1*H1*SIZE1) ; O/P - W2*H2*SIZE2 ; WT2 = (F2*F2*D1)*SIZE2
		self.W[2] = (self.W[1]-self.F[2]+2*self.P[2])/self.S[2] + 1
		self.H[2] = self.W[2]
		self.D[2] = self.SIZE[2]

		self.W[3] = (self.W[2]-self.F[3])/self.S[3] + 1
		self.H[3] = self.W[3]
		self.D[3] = self.SIZE[3]

		self.W[4] = self.H[4] = 1
		self.D[4] = self.SIZE[4]

		self.W[5] = self.H[5] = 1
		self.D[5] = self.SIZE[5]


	def train(training_dataset):

		#X - images ; Y - labels
		X = training_dataset[:,1]
		Y = training_dataset[:,0]

		self.forward_pass(X,Y)


	def forward_pass(images,labels):

		#Convolution layer 1


	def convolution_forward_pass(images,layer):

		# vectorize(images,F[layer])


}