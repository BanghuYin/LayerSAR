1--installing  
  1-1: OpenBLAS Installation  
	wget http://github.com/xianyi/OpenBLAS/archive/v0.2.20.tar.gz  
	tar -xzf SOpenBLAS-0.2.20.tar.gz  
	cd OpenBLAS-0.2.20  
	make  
	make PREFIX=/path/to/your/installation install  
  1-2: Downloading and Compiling  
	cd LayerSAR-MNIST  
	make  

2--running command:  
       ./layersar [INF] [network] [img_ind] [Timeout]  
	For example: ./layersar 5 models/mnist512.nnet 79  
	details of each parameter.  
	#1 parameter:INF, the total perturbation bound, 1 for default  
	#2 parameter:network dir  
	#3 parameter:img_ind, the image number for verification in images/  
	#4 parameter:Timeout; 600 (s) for default   
