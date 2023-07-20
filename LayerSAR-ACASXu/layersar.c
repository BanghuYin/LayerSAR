#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "split.h"

//export LD_LIBRARY_PATH=/opt/OpenBLAS/lib
//./layersar 1 ./nnet/ACASXU_run2a_1_1_batch_2000.nnet 0

int main(int argc, char *argv[]) {
	char *FULL_NET_PATH;

	int target = 0;

	if (argc > 9 || argc < 3) {
		printf("please specify a network\n");
		printf("./layersar [property] [network] "
				"[RELAX_MOD] [SPLITING_METHOD] "
				"[MAX_THREAD] [REFINE_delay_NUM]\n");
		exit(1);
	}

	for (int i = 1; i < argc; i++) {

		if (i == 1) {
			PROPERTY = atoi(argv[i]);
			if (PROPERTY < 0) {
				printf("No such property defined");
				exit(1);
			}
		}
		if (i == 2) {
			FULL_NET_PATH = argv[i];
		}
		if (i == 3) {
			RELAX_MOD = atoi(argv[i]);
		}
		if (i == 4) {
			SPLITING_METHOD = atoi(argv[i]);
		}
		if (i == 5) {
			MAX_THREAD = atoi(argv[i]);
		}
		if (i == 6) {
			REFINE_delay_NUM = atoi(argv[i]);
		}
	}
	openblas_set_num_threads(1);
	srand((unsigned) time(NULL));
	double time_spent;
	int i, j, layer;

	struct NNet* nnet = load_network(FULL_NET_PATH, target);

	int numLayers = nnet->numLayers;
	int inputSize = nnet->inputSize;
	int outputSize = nnet->outputSize;
	int maxLayerSize = nnet->maxLayerSize;

	float u[inputSize], l[inputSize];

	load_inputs(PROPERTY, inputSize, u, l);

	struct Matrix input_upper = { u, 1, nnet->inputSize };
	struct Matrix input_lower = { l, 1, nnet->inputSize };

	struct Interval input_interval = { input_lower, input_upper };


	if (DEBUG) {
		printf("input ranges before normalize:\n");
		printMatrix(&input_upper);
		printMatrix(&input_lower);
	}
	normalize_input_interval(nnet, &input_interval);

	float o[nnet->outputSize];
	struct Matrix output = { o, outputSize, 1 };
	memset(o, 0, sizeof(float) * nnet->outputSize);

	float o_upper[nnet->outputSize], o_lower[nnet->outputSize];
	struct Interval output_interval = { (struct Matrix ) { o_lower, outputSize,
					1 }, (struct Matrix ) { o_upper, outputSize, 1 } };

	int n = 0;
	int split_feature = -1;
	int depth = 0;
//	if (DEBUG)
    {
		printf("running property %d with network %s\n", PROPERTY,
				FULL_NET_PATH);
		printf("input ranges:\n");

		printMatrix(&input_upper);
		printMatrix(&input_lower);
	}
	for (int i = 0; i < inputSize; i++) {
		if (input_interval.upper_matrix.data[i]< input_interval.lower_matrix.data[i]) {
			printf("wrong input!\n");
			exit(0);
		}

		if (input_interval.upper_matrix.data[i]
				!= input_interval.lower_matrix.data[i]) {
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
	memset(new_equation_upper, 0,
			sizeof(float) *\
 (inputSize + 1) * maxLayerSize);
	memset(new_equation_lower, 0,
			sizeof(float) *\
 (inputSize + 1) * maxLayerSize);


	gettimeofday(&start, NULL);


	if (DEBUG) {
		printf("original output: ");
		printMatrix(&output);
	}


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

	forward_prop_value_symbl_linear2(nnet, &input_interval, &output_interval,
			 value_upper, value_lower, symbl_upper, symbl_lower, states,
			 equation_upper, equation_lower, new_equation_upper,new_equation_lower,
	   depth,1,0,1,0);

        printf("The output is: ");
			printMatrix(&output_interval.upper_matrix);
		        printMatrix(&output_interval.lower_matrix);


	result = check_functions(nnet, &output_interval);

	if (result == 0) //property is verified to be true!
			{
		result = check_property_by_forward_backward_refining(nnet,
				&input_interval,value_upper,
				value_lower, symbl_upper, symbl_lower, states,
				equation_upper, equation_lower, new_equation_upper,new_equation_lower,
				lp, &rule_num, PROPERTY, 1,depth,0,1,0);
	}

	if (result == 1 && adv_found==0)
	{	result=1;printf("Property is verified to be TRUE\n");}
	else if (result == -1 || adv_found==1)
	{	result=-1;printf("Property is verified to be FALSE\n");}
	else
	{	result=0;printf("Property is UNKNOWN\n");}
	gettimeofday(&finish, NULL);
	time_spent = ((float) (finish.tv_sec - start.tv_sec) * 1000000
			+ (float) (finish.tv_usec - start.tv_usec)) / 1000000;
        
	printf("network,result,forward_count,max_depth and time:%s %d %d %d %f\n",FULL_NET_PATH,result, forward_count, m_depth, time_spent);

	destroy_network(nnet);

	free(equation_upper);
	free(equation_lower);
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
