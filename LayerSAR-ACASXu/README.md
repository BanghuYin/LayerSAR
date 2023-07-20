1--installing
  1-1: OpenBLAS Installation
	wget http://github.com/xianyi/OpenBLAS/archive/v0.2.20.tar.gz
	tar -xzf SOpenBLAS-0.2.20.tar.gz
	cd OpenBLAS-0.2.20
	make
	make PREFIX=/path/to/your/installation install
  1-2: Downloading and Compiling
	cd LayerSAR-ACASXu
	make

2--running command:
	./layersar [property] [network_dir] [RELAX_MOD] [SPLITING_METHOD] [MAX_THREAD] [REFINE_delay_NUM]

	For example: ./layersar 1 ./acas_data/ACASXU_run2a_1_1_batch_2000.nnet

	details of each parameter.
	#1 parameter:property,1-4
	#2 parameter:network dir
	#3 parameter:RELAX_MOD=0;//0 for ours (default), 1 for POPL, 2 for Neurify.
	#4 parameter:SPLITING_METHOD=2; // 0 for dependency, 1 for smallest value range, 2 for largest value range(default), 3 for smallest index first, 4 for non_ful strategy
	#5 parameter:MAX_THREAD= 128;//1,4,16,64,128(default)
	#6 parameter:REFINE_delay_NUM=1;//1(default),2,3,4,5,6,7,8
