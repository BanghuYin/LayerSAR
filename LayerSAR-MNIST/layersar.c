#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "split.h"


//export LD_LIBRARY_PATH=/opt/OpenBLAS/lib
//./layersar 5 models/mnist512.nnet 79
int main( int argc, char *argv[]){

    char *FULL_NET_PATH;
    if(argc>6 || argc<3) {
        printf("./layersar [INF] [network] [img_ind] [Timeout]\n");
        exit(1);
    }
    int img_ind=0;
    for(int i=1;i<argc;i++){
        if(i==1){
            INF = atoi(argv[i]);
            if(INF<0){
                printf("Wrong INF");
                exit(1);
            }
        }
        if(i==2){
            FULL_NET_PATH = argv[i];
        }
        if(i==3){
        	img_ind = atoi(argv[i]);
        }
        if(i==4){
        	Timeout = atoi(argv[i]);
        }
    }
    openblas_set_num_threads(1);

    srand((unsigned)time(NULL));
    double time_spent;
    int i,j,layer;

    int image_start=0, image_length;

    int adv_num = 0;
    int non_adv = 0;
    int no_prove = 0;

    //image_length
//    for(int img_ind=0; img_ind<image_length;img_ind++){
//    for(int img_ind=0; img_ind<100;img_ind++){
     {
        int depth =0;
        int img = image_start + img_ind;
        adv_found=0;
        can_t_prove=0;
        forward_count=0;
        struct NNet* nnet = load_conv_network(FULL_NET_PATH, img);

        int numLayers    = nnet->numLayers;
        int inputSize    = nnet->inputSize;
        int outputSize   = nnet->outputSize;
        int maxLayerSize = nnet->maxLayerSize;

        float u[inputSize], l[inputSize], input_prev[inputSize];
        struct Matrix input_prev_matrix = {input_prev, 1, inputSize};

        struct Matrix input_upper = {u,1,nnet->inputSize};
        struct Matrix input_lower = {l,1,nnet->inputSize};
        struct Interval input_interval = {input_lower, input_upper};

        initialize_input_interval(nnet, img, inputSize, input_prev, u, l);
        if(PROPERTY<500){
            normalize_input(nnet, &input_prev_matrix);
            normalize_input_interval(nnet, &input_interval);
        }
        float grad_upper[inputSize], grad_lower[inputSize];
        struct Interval grad_interval = {(struct Matrix){grad_upper, 1, inputSize},
                                         (struct Matrix){grad_lower, 1, inputSize}};

        float o[nnet->outputSize];
        struct Matrix output = {o, outputSize, 1};

        float o_upper[nnet->outputSize], o_lower[nnet->outputSize];
        struct Interval output_interval = {(struct Matrix){o_lower, outputSize, 1},
                                           (struct Matrix){o_upper, outputSize, 1}};

        int n = 0;
        int split_feature = -1;
        printf("running image %d with network %s\n", img, FULL_NET_PATH);
//        printf("Infinite Norm: %f\n", INF);
        //printMatrix(&input_upper);
        //printMatrix(&input_lower);
        for(int i=0;i<inputSize;i++){
            if(input_interval.upper_matrix.data[i]<input_interval.lower_matrix.data[i]){
                printf("wrong input!\n");
                exit(0);
            }
            if(input_interval.upper_matrix.data[i]!=input_interval.lower_matrix.data[i]){
                n++;
            }
        }
	gettimeofday(&start, NULL);
	int isOverlap = 0;

	//Interval values
	float **value_upper = (float**) malloc(sizeof(float*) * (numLayers + 1));
	for (int i = 0; i < numLayers + 1; i++) {
		value_upper[i] = (float*) malloc(sizeof(float) * maxLayerSize);
		memset(value_upper[i], 0, sizeof(float) * maxLayerSize);
	}

	float **value_lower = (float**) malloc(sizeof(float*) * (numLayers + 1));
	for (int i = 0; i < numLayers + 1; i++) {
		value_lower[i] = (float*) malloc(sizeof(float) * maxLayerSize);
		memset(value_lower[i], 0, sizeof(float) * maxLayerSize);
	}

	float **symbl_upper = (float**) malloc(sizeof(float*) * (numLayers + 1));
	for (int i = 0; i < numLayers + 1; i++) {
		symbl_upper[i] = (float*) malloc(
				sizeof(float) * maxLayerSize * (inputSize + 1));
		memset(symbl_upper[i], 0,
				sizeof(float) * maxLayerSize * (inputSize + 1));
	}

	float **symbl_lower = (float**) malloc(sizeof(float*) * (numLayers + 1));
	for (int i = 0; i < numLayers + 1; i++) {
		symbl_lower[i] = (float*) malloc(
				sizeof(float) * maxLayerSize * (inputSize + 1));
		memset(symbl_lower[i], 0,
				sizeof(float) * maxLayerSize * (inputSize + 1));
	}

	int **states = (int **) malloc(sizeof(int*) * (numLayers + 1));
	for (int i = 0; i < numLayers + 1; i++) {
		states[i] = (int*) malloc(sizeof(int) * maxLayerSize);
		memset(states[i], -1, sizeof(int) * maxLayerSize);
	}


	float *equation_upper = (float*) malloc(
			sizeof(float) *\
 (inputSize + 1) * maxLayerSize);
	float *equation_lower = (float*) malloc(
			sizeof(float) *\
 (inputSize + 1) * maxLayerSize);

	float *equation_conv_upper = (float*) malloc(
				sizeof(float) *\
	 (inputSize + 1) * maxLayerSize);
		float *equation_conv_lower = (float*) malloc(
				sizeof(float) *\
	 (inputSize + 1) * maxLayerSize);

	float *new_equation_upper = (float*) malloc(
			sizeof(float) *\
 (inputSize + 1) * maxLayerSize);
	float *new_equation_lower = (float*) malloc(
			sizeof(float) *\
 (inputSize + 1) * maxLayerSize);

	float *equation_input_lower = (float*) malloc(
			sizeof(float) *\
 (inputSize + 1) * nnet->layerSizes[1]);
	float *equation_input_upper = (float*) malloc(
			sizeof(float) *\
 (inputSize + 1) * nnet->layerSizes[1]);

	memset(equation_upper, 0, sizeof(float) *\
 (inputSize + 1) * maxLayerSize);
	memset(equation_lower, 0, sizeof(float) *\
 (inputSize + 1) * maxLayerSize);



	memset(equation_conv_upper, 0, sizeof(float) *\
 (inputSize + 1) * maxLayerSize);
	memset(equation_conv_lower, 0, sizeof(float) *\
 (inputSize + 1) * maxLayerSize);


	memset(new_equation_upper, 0,
			sizeof(float) *\
 (inputSize + 1) * maxLayerSize);

	memset(new_equation_lower, 0,
			sizeof(float) *\
 (inputSize + 1) * maxLayerSize);

	struct Interval equation_interval = {
			(struct Matrix ) { (float*) equation_lower, inputSize + 1,
							nnet->layerSizes[1] }, (struct Matrix ) {
									(float*) equation_upper, inputSize + 1,
									nnet->layerSizes[1] } };


	gettimeofday(&start, NULL);

	if (DEBUG) {
		printf("original output: ");
		printMatrix(&output);
	}

	 evaluate_conv(nnet, &input_prev_matrix, &output);

	lprec *lp;

	int rule_num = 0;
	int Ncol = inputSize;
	REAL row[Ncol + 1];
	lp = make_lp(0, Ncol);

	set_verbose(lp, IMPORTANT);
	set_input_constraints(&input_interval, lp, &rule_num);
	int result = 0;

	int res_temp;

	float output_lower_upper[nnet->outputSize], output_upper_lower[nnet->outputSize];

    int output_map_o[outputSize];
    for(int oi=0;oi<outputSize;oi++){
    	  if(oi!=nnet->target)
    	output_map_o[oi]=1;
    	  else
    		  output_map_o[oi]=0;
    }

	forward_prop_value_symbl_linear2(nnet, &input_interval, &output_interval,
				 value_upper, value_lower,output_lower_upper,output_upper_lower, symbl_upper, symbl_lower, states,
		   depth,1,0,1,0,output_map_o,lp,&rule_num);


     for(int oi=0;oi<outputSize;oi++){
    	 if(output_map_o==1)
    	 {
			 if(output_interval.upper_matrix.data[oi]>output_interval.lower_matrix.data[nnet->target] && oi!=nnet->target){
				 output_map_o[oi]=1;
			 }
			 else{
				 output_map_o[oi]=0;
			 }
    	 }
     }

    printf("Target is %d,One shot approximation:\n",nnet->target);
    printf("upper_matrix:");
    printMatrix(&output_interval.upper_matrix);
    printf("lower matrix:");
    printMatrix(&output_interval.lower_matrix);

    int cont_r=0,det_1=0;
     for(int i=0;i<numLayers;i++)
     	for(int j=0;j<maxLayerSize;j++)
     	{
     		if(states[i][j]==0||states[i][j]==2)
     		{
     			cont_r+=1;
     		}
     	}
     printf("cont_r is %d\n",cont_r);

    if(adv_found==1)
    	result = -1;
    else
	    result = check_functions_norm(nnet, &output_interval);

	if (result == 0) //property is verified to be true!
			{
		result = check_property_by_forward_backward_refining(nnet,\
						&input_interval, &output_interval,value_upper,\
						value_lower, symbl_upper, symbl_lower, states,\
						lp, &rule_num, PROPERTY, 1,depth,0,1,0,output_map_o);
	}

    if(adv_found==1)
    	result = -1;

	if (result == 1 && can_t_prove==0)
	{
		printf("Property is verified to be TRUE\n");
	}
	else if (result == -1)
	{
		printf("Property is verified to be FALSE\n");
	}
	else
	{
		printf("Property is UNKNOWN\n");
		result=0;
     //  no_prove ++;
	}

	gettimeofday(&finish, NULL);
	time_spent = ((float) (finish.tv_sec - start.tv_sec) * 1000000
			+ (float) (finish.tv_usec - start.tv_usec)) / 1000000;
	 printf("INF,image,result,forward_count,max_depth and time:%f %d %d %d %f\n",INF,img_ind,result, forward_count, time_spent);

	 destroy_conv_network(nnet);

	free(equation_upper);
	free(equation_lower);
	free(equation_conv_upper);
	free(equation_conv_lower);
	free(new_equation_upper);
	free(new_equation_lower);
	free(equation_input_upper);
	free(equation_input_lower);
	delete_lp(lp);

	for (int i = 0; i < numLayers + 1; i++) {
		free(value_upper[i]);
	}
	free(value_upper);

	for (int i = 0; i < numLayers + 1; i++) {
		free(value_lower[i]);
	}
	free(value_lower);

	for (int i = 0; i < numLayers + 1; i++) {
		free(states[i]);
	}
	free(states);

	for (int i = 0; i < numLayers + 1; i++) {
		free(symbl_upper[i]);
	}
	free(symbl_upper);

	for (int i = 0; i < numLayers + 1; i++) {
		free(symbl_lower[i]);
	}
	free(symbl_lower);
    }
}
