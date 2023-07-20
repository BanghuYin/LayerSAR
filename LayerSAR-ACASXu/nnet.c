#include "nnet.h"
#include "split.h"

int PROPERTY = 5;
char *LOG_FILE = "logs/log.txt";
FILE *fp;

/*
 * Load_network is a function modified from Reluplex
 * It takes in a nnet filename with path and load the 
 * network from the file
 * Outputs the NNet instance of loaded network.
 */
struct NNet *load_network(const char* filename, int target)
{

    FILE *fstream = fopen(filename,"r");

    if (fstream == NULL) {
        printf("Wrong network!\n");
        exit(1);
    }

    int bufferSize = 10240;
    char *buffer = (char*)malloc(sizeof(char)*bufferSize);
    char *record, *line;
    int i=0, layer=0, row=0, j=0, param=0;

    struct NNet *nnet = (struct NNet*)malloc(sizeof(struct NNet));

    nnet->target = target;

    line=fgets(buffer,bufferSize,fstream);

    while (strstr(line, "//") != NULL) {
        line = fgets(buffer,bufferSize,fstream); 
    }

    record = strtok(line,",\n");
    nnet->numLayers = atoi(record);
    nnet->inputSize = atoi(strtok(NULL,",\n"));
    nnet->outputSize = atoi(strtok(NULL,",\n"));
    nnet->maxLayerSize = atoi(strtok(NULL,",\n"));

    if(nnet->inputSize>nnet->maxLayerSize)
	nnet->maxLayerSize=nnet->inputSize;

    nnet->layerSizes = (int*)malloc(sizeof(int)*(nnet->numLayers+1));
    line = fgets(buffer,bufferSize,fstream);
    record = strtok(line,",\n");

//    printf("numLayers is %d",nnet->numLayers);
    fflush(stdout);
    for (i = 0;i<((nnet->numLayers)+1);i++) {
        nnet->layerSizes[i] = atoi(record);
        printf("layersize is %d\n",nnet->layerSizes[i]);
          fflush(stdout);
        record = strtok(NULL,",\n");
    }
//    printf("after layersize loading\n");
//             fflush(stdout);
    line = fgets(buffer,bufferSize,fstream);
    record = strtok(line,",\n");
    nnet->symmetric = atoi(record);

    nnet->mins = (float*)malloc(sizeof(float)*nnet->inputSize);
    line = fgets(buffer,bufferSize,fstream);
    record = strtok(line,",\n");

    for (i = 0;i<(nnet->inputSize);i++) {
        nnet->mins[i] = (float)atof(record);
        record = strtok(NULL,",\n");
    }
    nnet->maxes = (float*)malloc(sizeof(float)*nnet->inputSize);
    line = fgets(buffer,bufferSize,fstream);
    record = strtok(line,",\n");

    for (i = 0;i<(nnet->inputSize);i++) {
        nnet->maxes[i] = (float)atof(record);
        record = strtok(NULL,",\n");
    }
    nnet->means = (float*)malloc(sizeof(float)*(nnet->inputSize+1));
    line = fgets(buffer,bufferSize,fstream);
    record = strtok(line,",\n");

    for (i = 0;i<((nnet->inputSize)+1);i++) { //means为什么是inputSize+1个？？
        nnet->means[i] = (float)atof(record);
        record = strtok(NULL,",\n");
    }

    nnet->ranges = (float*)malloc(sizeof(float)*(nnet->inputSize+1));
    line = fgets(buffer,bufferSize,fstream);
    record = strtok(line,",\n");

    for (i = 0;i<((nnet->inputSize)+1);i++) { //ranges为什么是inputSize+1个？？
        nnet->ranges[i] = (float)atof(record);
        record = strtok(NULL,",\n");
    }

    nnet->matrix = (float****)malloc(sizeof(float *)*nnet->numLayers);

    for (layer = 0;layer<(nnet->numLayers);layer++) {
        nnet->matrix[layer] =\
                (float***)malloc(sizeof(float *)*2);
        nnet->matrix[layer][0] =\
                (float**)malloc(sizeof(float *)*nnet->layerSizes[layer+1]);
        nnet->matrix[layer][1] =\
                (float**)malloc(sizeof(float *)*nnet->layerSizes[layer+1]);

        for (row = 0;row<nnet->layerSizes[layer+1];row++) {
            nnet->matrix[layer][0][row] =\
                    (float*)malloc(sizeof(float)*nnet->layerSizes[layer]);
            nnet->matrix[layer][1][row] = (float*)malloc(sizeof(float));
        }

    }
    
    layer = 0;
    param = 0;
    i=0;
    j=0;

    char *tmpptr=NULL;

    float w = 0.0;

    while ((line = fgets(buffer,bufferSize,fstream)) != NULL) {

        if (i >= nnet->layerSizes[layer+1]) {

            if (param==0) {
                param = 1;
            }
            else {
                param = 0;
                layer++;
            }

            i=0;
            j=0;
        }

        record = strtok_r(line,",\n", &tmpptr);

        while (record != NULL) {   
            w = (float)atof(record);
            nnet->matrix[layer][param][i][j] = w;
            j++;
            record = strtok_r(NULL, ",\n", &tmpptr);
        }

        tmpptr=NULL;
        j=0;
        i++;
    }

    float orig_weights[nnet->maxLayerSize];
    float orig_bias;

    struct Matrix *weights=malloc(nnet->numLayers*sizeof(struct Matrix));
    struct Matrix *bias = malloc(nnet->numLayers*sizeof(struct Matrix));

    for (int layer=0;layer<nnet->numLayers;layer++) {
        weights[layer].row = nnet->layerSizes[layer];
        weights[layer].col = nnet->layerSizes[layer+1];
        weights[layer].data = (float*)malloc(sizeof(float)\
                    * weights[layer].row * weights[layer].col);

        int n=0;

        if (PROPERTY != 1 && PROPERTY != 7) {

            /* weights in the last layer minus the weights of true label output. */
            if (layer == nnet->numLayers-1) {
                orig_bias = nnet->matrix[layer][1][nnet->target][0];
                memcpy(orig_weights, nnet->matrix[layer][0][nnet->target],\
                            sizeof(float)*nnet->maxLayerSize);

                for (int i=0;i<weights[layer].col;i++) {

                    for (int j=0;j<weights[layer].row;j++) {
                        weights[layer].data[n] =\
                                nnet->matrix[layer][0][i][j]-orig_weights[j];
                        n++;
                    }

                }

                bias[layer].col = nnet->layerSizes[layer+1];
                bias[layer].row = (float)1;
                bias[layer].data = (float*)malloc(sizeof(float)*bias[layer].col);

                for (int i=0;i<bias[layer].col;i++) {
                    bias[layer].data[i] = nnet->matrix[layer][1][i][0]-orig_bias;
                }
            }
            else {

                for (int i=0;i<weights[layer].col;i++) {

                    for (int j=0;j<weights[layer].row;j++) {
                        weights[layer].data[n] = nnet->matrix[layer][0][i][j];
                        n++;
                    }

                }

                bias[layer].col = nnet->layerSizes[layer+1];
                bias[layer].row = (float)1;
                bias[layer].data = (float*)malloc(sizeof(float) *\
                                        bias[layer].col);

                for (int i=0;i<bias[layer].col;i++) {
                    bias[layer].data[i] = nnet->matrix[layer][1][i][0];
                }

            }
        }
        else {

            for (int i=0;i<weights[layer].col;i++) {

                for (int j=0;j<weights[layer].row;j++) {
                    weights[layer].data[n] = nnet->matrix[layer][0][i][j];
                    n++;
                }

            }

            bias[layer].col = nnet->layerSizes[layer+1];
            bias[layer].row = (float)1;
            bias[layer].data = (float*)malloc(sizeof(float)*bias[layer].col);

            for (int i=0;i<bias[layer].col;i++) {
                bias[layer].data[i] = nnet->matrix[layer][1][i][0];
            }

        }

    }

    nnet->weights = weights;
    nnet->bias = bias;
    if(DEBUG)
    {
		printf("\nweights 0:\n");
		printMatrix(&weights[0]);
		printf("bias 0:\n");
		printMatrix(&bias[0]);

		printf("weights 1:\n");
		printMatrix(&weights[1]);
		printf("bias 1:\n");
		printMatrix(&bias[1]);
    }

    free(buffer);
    fclose(fstream);

    return nnet;

}


/*
 * destroy_network is a function modified from Reluplex
 * It release all the memory mallocated to the network instance
 * It takes in the instance of nnet
 */
void destroy_network(struct NNet *nnet)
{

    int i=0, row=0;
    if (nnet != NULL) {

        for (i=0;i<(nnet->numLayers);i++) {

            for (row=0;row<nnet->layerSizes[i+1];row++) {
                free(nnet->matrix[i][0][row]);
                free(nnet->matrix[i][1][row]);
            }

            free(nnet->matrix[i][0]);
            free(nnet->matrix[i][1]);
            free(nnet->weights[i].data);
            free(nnet->bias[i].data);
            free(nnet->matrix[i]);
        }

        free(nnet->weights);
        free(nnet->bias);
        free(nnet->layerSizes);
        free(nnet->mins);
        free(nnet->maxes);
        free(nnet->means);
        free(nnet->ranges);
        free(nnet->matrix);
        free(nnet);
    }

}


/*
 * Load the inputs of all the predefined properties
 * It takes in the property and input pointers
 */
void load_inputs(int PROPERTY, int inputSize, float *u, float *l)
{

    if (PROPERTY == 1) {
        float upper[] = {60760,3.141592,3.141592,1200,60};
        float lower[] = {55947.691,-3.141592,-3.141592,1145,0};
        memcpy(u, upper, sizeof(float)*inputSize);
        memcpy(l, lower, sizeof(float)*inputSize);
    }

    if (PROPERTY == 2) {
        float upper[] = {60760,3.141592,3.141592, 1200, 60};
        float lower[] = {55947.691,-3.141592,-3.141592,1145,0};
        memcpy(u, upper, sizeof(float)*inputSize);
        memcpy(l, lower, sizeof(float)*inputSize);
    }

    if (PROPERTY == 3) {
        float upper[] = {1800,0.06,3.141592,1200,1200};
        float lower[] = {1500,-0.06,3.10,980,960};
        memcpy(u, upper, sizeof(float)*inputSize);
        memcpy(l, lower, sizeof(float)*inputSize);
    }

    if (PROPERTY == 4) {
        float upper[] = {1800,0.06,0,1200,800};
        float lower[] = {1500,-0.06,0,1000,700};
        memcpy(u, upper, sizeof(float)*inputSize);
        memcpy(l, lower, sizeof(float)*inputSize);
    }

    if (PROPERTY == 5) {
        float upper[] = {400,0.4,-3.1415926+0.005,400,400};
        float lower[] = {250,0.2,-3.1415926,100,0};
        memcpy(u, upper, sizeof(float)*inputSize);
        memcpy(l, lower, sizeof(float)*inputSize);
    }

    if (PROPERTY==16) {
        float upper[] = {62000,-0.7,-3.141592+0.005,200,1200};
        float lower[] = {12000,-3.141592,-3.141592,100,0};
        memcpy(u, upper, sizeof(float)*inputSize);
        memcpy(l, lower, sizeof(float)*inputSize);
    }

    if (PROPERTY==26) {
        float upper[] = {62000,3.141592,-3.141592+0.005,200,1200};
        float lower[] = {12000,0.7,-3.141592,100,0};
        memcpy(u, upper, sizeof(float)*inputSize);
        memcpy(l, lower, sizeof(float)*inputSize);
    }

    if (PROPERTY==7) {
        float upper[] = {60760,3.141592,3.141592,1200,1200};
        float lower[] = {0,-3.141592,-3.141592,100,0};
        memcpy(u, upper, sizeof(float)*inputSize);
        memcpy(l, lower, sizeof(float)*inputSize);
    }

    if (PROPERTY==8) {
        float upper[] = {60760,-3.141592*0.75,0.1,1200,1200};
        float lower[] = {0,-3.141592,-0.1,600,600};
        memcpy(u, upper, sizeof(float)*inputSize);
        memcpy(l, lower, sizeof(float)*inputSize);
    }

    if (PROPERTY==9) {
        float upper[] = {7000,-0.14,-3.141592+0.01,150,150};
        float lower[] = {2000,-0.4,-3.141592,100,0};
        memcpy(u, upper, sizeof(float)*inputSize);
        memcpy(l, lower, sizeof(float)*inputSize);
    }

    if (PROPERTY==10) {
        float upper[] = {60760,3.141592,-3.141592+0.01,1200,1200};
        float lower[] = {36000,0.7,-3.141592,900,600};
        memcpy(u, upper, sizeof(float)*inputSize);
        memcpy(l, lower, sizeof(float)*inputSize);
    }

    if (PROPERTY == 11) {
        float upper[] = {400,0.4,-3.1415926+0.005,400,400};
        float lower[] = {250,0.2,-3.1415926,100,0};
        memcpy(u, upper, sizeof(float)*inputSize);
        memcpy(l, lower, sizeof(float)*inputSize);
    }

    if (PROPERTY == 12) {
        float upper[] = {60760,3.141592,3.141592, 1200, 60};
        float lower[] = {55947.691,-3.141592,-3.141592,1145,0};
        memcpy(u, upper, sizeof(float)*inputSize);
        memcpy(l, lower, sizeof(float)*inputSize);
    }

    if (PROPERTY == 13) {
        float upper[] = {60760,3.141592,3.141592, 360, 360};
        float lower[] = {60000,-3.141592,-3.141592,0,0};
        memcpy(u, upper, sizeof(float)*inputSize);
        memcpy(l, lower, sizeof(float)*inputSize);
    }

    if (PROPERTY == 14) {
        float upper[] = {400,0.4,-3.1415926+0.005,400,400};
        float lower[] = {250,0.2,-3.1415926,100,0};
        memcpy(u, upper, sizeof(float)*inputSize);
        memcpy(l, lower, sizeof(float)*inputSize);
    }

    if (PROPERTY == 15) {
        float upper[] = {400,-0.2,-3.1415926+0.005,400,400};
        float lower[] = {250,-0.4,-3.1415926,100,0};
        memcpy(u, upper, sizeof(float)*inputSize);
        memcpy(l, lower, sizeof(float)*inputSize);
    }

    if (PROPERTY == 100) {
        float upper[] = {400,0,-3.1415926+0.025,250,200};
        float lower[] = {250,0,-3.1415926+0.025,250,200};
        memcpy(u, upper, sizeof(float)*inputSize);
        memcpy(l, lower, sizeof(float)*inputSize);
    }

    if (PROPERTY == 101) {
        float upper[] = {400,0.4,-3.1415926+0.025,250,200};
        float lower[] = {250,0.2,-3.1415926+0.025,250,200};
        memcpy(u, upper, sizeof(float)*inputSize);
        memcpy(l, lower, sizeof(float)*inputSize);
    }

    if (PROPERTY == 102) {
        float upper[] = {400,0.2,-3.1415926+0.05,0,0};
        float lower[] = {250,0.2,-3.1415926+0.05,0,0};
        memcpy(u, upper, sizeof(float)*inputSize);
        memcpy(l, lower, sizeof(float)*inputSize);
    }

    if (PROPERTY == 110) {
        float upper[] = {10000,3.141592,-3.141592+0.01,1200,1200};
        float lower[] = {1000,3.141592,-3.141592+0.01,1200,1200};
        memcpy(u, upper, sizeof(float)*inputSize);
        memcpy(l, lower, sizeof(float)*inputSize);
    }

    if (PROPERTY == 111) {
        float upper[] = {1000,3.141592,-3.141592+0.01,1200,1200};
        float lower[] = {1000,3.141592,-3.141592+0.01,0,1200};
        memcpy(u, upper, sizeof(float)*inputSize);
        memcpy(l, lower, sizeof(float)*inputSize);
    }

    if (PROPERTY == 112) {
        float upper[] = {1000,3.141592,-3.141592+0.01,1200,1200};
        float lower[] = {1000,3.141592,-3.141592+0.01,1200,0};
        memcpy(u, upper, sizeof(float)*inputSize);
        memcpy(l, lower, sizeof(float)*inputSize);
    }

}


/*
 * Following functions denomalize and normalize the concrete inputs
 * and input intervals.
 * They take in concrete inputs or input intervals.
 * Output normalized or denormalized concrete inputs or input intervals.
 */
void denormalize_input(struct NNet *nnet, struct Matrix *input)
{

    for (int i=0; i<nnet->inputSize;i++) {
        input->data[i] = input->data[i]*(nnet->ranges[i]) + nnet->means[i];
    }

}
void denormalize_input_interval(struct NNet *nnet, struct Interval *input)
{

    denormalize_input(nnet, &input->upper_matrix);
    denormalize_input(nnet, &input->lower_matrix);

}

void normalize_input(struct NNet *nnet, struct Matrix *input)
{

    for (int i=0;i<nnet->inputSize;i++) {

        if (input->data[i] > nnet->maxes[i]) {
            input->data[i] = (nnet->maxes[i]-nnet->means[i])/(nnet->ranges[i]);
        }
        else if (input->data[i] < nnet->mins[i]) {
            input->data[i] = (nnet->mins[i]-nnet->means[i])/(nnet->ranges[i]);
        }
        else {
            input->data[i] = (input->data[i]-nnet->means[i])/(nnet->ranges[i]);
        }
    }
}


void normalize_input_interval(struct NNet *nnet, struct Interval *input)
{
    normalize_input(nnet, &input->upper_matrix);
    normalize_input(nnet, &input->lower_matrix);
}


void set_input_constraints(struct Interval *input, lprec *lp, int *rule_num){
    int Ncol = input->upper_matrix.col;
    for(int var=1;var<Ncol+1;var++){
        set_bounds(lp,var,input->lower_matrix.data[var-1],input->upper_matrix.data[var-1]);
    }
}


int forward_prop_value_symbl_linear2(struct NNet *nnet, struct Interval *input,
                                     struct Interval *output,
                                     float **value_upper, float **value_lower,
                                  //   float *output_lower_upper, float *output_upper_lower,
                                     float **symbol_upper, float **symbol_lower, int **states,
                                     float *equation_upper, float *equation_lower,
                                     float *new_equation_upper, float *new_equation_lower,
                                     int depth,int is_first_time,int split_layer_pre,int change_split_layer,int input_refined)
{

    pthread_mutex_lock(&lock);
    forward_count++;
    pthread_mutex_unlock(&lock);
    int i,j,k,layer;

    int numLayers    = nnet->numLayers;
    int inputSize    = nnet->inputSize;
    int outputSize   = nnet->outputSize;
    int maxLayerSize = nnet->maxLayerSize;

    memset(equation_upper,0,sizeof(float)*(inputSize+1)*maxLayerSize);
    memset(equation_lower,0,sizeof(float)*(inputSize+1)*maxLayerSize);

    struct Interval equation_inteval = {
            (struct Matrix){(float*)equation_lower, inputSize+1, inputSize},
            (struct Matrix){(float*)equation_upper, inputSize+1, inputSize}
    };
    struct Interval new_equation_inteval = {
            (struct Matrix){(float*)new_equation_lower, inputSize+1, maxLayerSize},
            (struct Matrix){(float*)new_equation_upper, inputSize+1, maxLayerSize}
    };

    float tempVal_upper=0.0, tempVal_lower=0.0;
    float upper_s_lower=0.0, lower_s_upper=0.0;

    for (i=0; i < nnet->inputSize; i++)
    {
        equation_lower[i*(inputSize+1)+i] = 1;
        equation_upper[i*(inputSize+1)+i] = 1;
    }


    for (layer = 0; layer<(numLayers); layer++)
    {
    	struct Matrix bias = nnet->bias[layer];
        if(layer!=0) memset(new_equation_upper, 0, sizeof(float)*(inputSize+1)*maxLayerSize);
        if(layer!=0) memset(new_equation_lower, 0, sizeof(float)*(inputSize+1)*maxLayerSize);

        if(layer<split_layer_pre && is_first_time==0 && change_split_layer==0)
        {
            for (i=0; i < nnet->layerSizes[layer+1]; i++)
            {
				for(k=0;k<inputSize+1;k++){
					 new_equation_upper[k+i*(inputSize+1)] = symbol_upper[layer+1][k+i*(inputSize+1)];
					 new_equation_lower[k+i*(inputSize+1)]= symbol_lower[layer+1][k+i*(inputSize+1)];
				}
            }
        }
        else{
         	struct Matrix weights = nnet->weights[layer];
   	        float p[weights.col*weights.row];
   	        float n[weights.col*weights.row];
   	        memset(p, 0, sizeof(float)*weights.col*weights.row);
   	        memset(n, 0, sizeof(float)*weights.col*weights.row);
   	        struct Matrix pos_weights = {p, weights.row, weights.col};
   	        struct Matrix neg_weights = {n, weights.row, weights.col};
   	        for(i=0;i<weights.row*weights.col;i++){
   	            if(weights.data[i]>=0){
   	                p[i] = weights.data[i];
   	            }
   	            else{
   	                n[i] = weights.data[i];
   	            }
   	        }
   	    pthread_mutex_lock(&lock);
        matmul(&equation_inteval.upper_matrix, &pos_weights, &new_equation_inteval.upper_matrix);
        matmul_with_bias(&equation_inteval.lower_matrix, &neg_weights, &new_equation_inteval.upper_matrix);

        matmul(&equation_inteval.lower_matrix, &pos_weights, &new_equation_inteval.lower_matrix);
        matmul_with_bias(&equation_inteval.upper_matrix, &neg_weights, &new_equation_inteval.lower_matrix);
        pthread_mutex_unlock(&lock);
        for (i=0; i < nnet->layerSizes[layer+1]; i++)
        {

            new_equation_lower[inputSize+i*(inputSize+1)] += bias.data[i];
            new_equation_upper[inputSize+i*(inputSize+1)] += bias.data[i];


            for(k=0;k<inputSize+1;k++){
              	symbol_upper[layer+1][k+i*(inputSize+1)] = new_equation_upper[k+i*(inputSize+1)];
                symbol_lower[layer+1][k+i*(inputSize+1)] = new_equation_lower[k+i*(inputSize+1)];
            }
        }
        }
        for (i=0; i < nnet->layerSizes[layer+1]; i++)
        {
         if(layer<(numLayers-1)){

           if(states[layer+1][i]==0)
                {
                	 for(k=0;k<inputSize+1;k++){
                	                        new_equation_upper[k+i*(inputSize+1)] = 0;
                	                        new_equation_lower[k+i*(inputSize+1)] = 0;
                	                    }
                	 states[layer+1][i] = 0;

                }
                else if(states[layer+1][i]==2)
                {

                	states[layer+1][i] = 2;
                }
                else
                {
               	   {
               		   tempVal_upper = tempVal_lower = 0.0;
               		   lower_s_upper = upper_s_lower = 0.0;

               		   if(NEED_OUTWARD_ROUND){
               			   for(k=0;k<inputSize;k++){
               				   if(new_equation_lower[k+i*(inputSize+1)]>=0){
									tempVal_lower += new_equation_lower[k+i*(inputSize+1)] * input->lower_matrix.data[k]-OUTWARD_ROUND;
									lower_s_upper += new_equation_lower[k+i*(inputSize+1)] * input->upper_matrix.data[k]-OUTWARD_ROUND;
               				   }
               				   else{
									tempVal_lower += new_equation_lower[k+i*(inputSize+1)] * input->upper_matrix.data[k]-OUTWARD_ROUND;
									lower_s_upper += new_equation_lower[k+i*(inputSize+1)] * input->lower_matrix.data[k]-OUTWARD_ROUND;
               				   }
               				   if(new_equation_upper[k+i*(inputSize+1)]>=0){
									tempVal_upper += new_equation_upper[k+i*(inputSize+1)] * input->upper_matrix.data[k]+OUTWARD_ROUND;
									upper_s_lower += new_equation_upper[k+i*(inputSize+1)] * input->lower_matrix.data[k]+OUTWARD_ROUND;
               				   }
               				   else{
								tempVal_upper += new_equation_upper[k+i*(inputSize+1)] * input->lower_matrix.data[k]+OUTWARD_ROUND;
								upper_s_lower += new_equation_upper[k+i*(inputSize+1)] * input->upper_matrix.data[k]+OUTWARD_ROUND;
               				   }
               			   }
               		   }
               		   else{
						for(k=0;k<inputSize;k++){
							if(layer==0){
								if(new_equation_lower[k+i*(inputSize+1)]!=new_equation_upper[k+i*(inputSize+1)]){
									printf("wrong!\n");
								}
							}
							if(new_equation_lower[k+i*(inputSize+1)]>=0){
								tempVal_lower = tempVal_lower + new_equation_lower[k+i*(inputSize+1)] * input->lower_matrix.data[k];
								lower_s_upper =lower_s_upper+ new_equation_lower[k+i*(inputSize+1)] * input->upper_matrix.data[k];
							}
							else{
								tempVal_lower = tempVal_lower + new_equation_lower[k+i*(inputSize+1)] * input->upper_matrix.data[k];
								lower_s_upper =lower_s_upper+ new_equation_lower[k+i*(inputSize+1)] * input->lower_matrix.data[k];
							}
							if(new_equation_upper[k+i*(inputSize+1)]>=0){
								tempVal_upper = tempVal_upper + new_equation_upper[k+i*(inputSize+1)] * input->upper_matrix.data[k];
								upper_s_lower = upper_s_lower + new_equation_upper[k+i*(inputSize+1)] * input->lower_matrix.data[k];
							}
							else{
								tempVal_upper = tempVal_upper + new_equation_upper[k+i*(inputSize+1)] * input->lower_matrix.data[k];
								upper_s_lower = upper_s_lower + new_equation_upper[k+i*(inputSize+1)] * input->upper_matrix.data[k];
							}
                }
            }
            tempVal_lower =tempVal_lower + new_equation_lower[inputSize+i*(inputSize+1)];
            lower_s_upper =lower_s_upper + new_equation_lower[inputSize+i*(inputSize+1)];
            tempVal_upper = tempVal_upper+new_equation_upper[inputSize+i*(inputSize+1)];
            upper_s_lower = upper_s_lower + new_equation_upper[inputSize+i*(inputSize+1)];


			if(is_first_time==0)
			{
				if(value_lower[layer+1][i]>tempVal_lower)
					tempVal_lower=value_lower[layer+1][i];
				else
				    value_lower[layer+1][i]=tempVal_lower;
				if(value_upper[layer+1][i]>tempVal_upper)
				    value_upper[layer+1][i]=tempVal_upper;
				else
					tempVal_upper=value_upper[layer+1][i];
			}
			else
			{
				value_lower[layer+1][i]=tempVal_lower;
				value_upper[layer+1][i]=tempVal_upper;
			}
		}

			if (tempVal_upper<=0.0){
				for(k=0;k<inputSize+1;k++){
					new_equation_upper[k+i*(inputSize+1)] = 0;
					new_equation_lower[k+i*(inputSize+1)] = 0;
				}
				states[layer+1][i] = 0;
			}
			else if(tempVal_lower>=0.0){
				states[layer+1][i] = 2;
			}
			else{

				if(RELAX_MOD==0)
				{

					if(upper_s_lower<0.0){
						for(k=0;k<inputSize+1;k++){
							new_equation_upper[k+i*(inputSize+1)] =\
													new_equation_upper[k+i*(inputSize+1)]*\
													tempVal_upper / (tempVal_upper-upper_s_lower);
						}
						new_equation_upper[inputSize+i*(inputSize+1)] -= tempVal_upper*upper_s_lower/\
															(tempVal_upper-upper_s_lower);
					}


					if(lower_s_upper<0.0){
						for(k=0;k<inputSize+1;k++){
							new_equation_lower[k+i*(inputSize+1)] = 0;
						}
					}
					else
					{
						if(-tempVal_lower>lower_s_upper)
						{
							 for(k=0;k<inputSize+1;k++){
							   new_equation_lower[k+i*(inputSize+1)] = 0;
							  }
						}
						else
						{

						}
					}
					states[layer+1][i] = 1;
				}


				if(RELAX_MOD==1)
				{
				   for(k=0;k<inputSize+1;k++)
			  {
						  new_equation_upper[k+i*(inputSize+1)] =\
											new_equation_upper[k+i*(inputSize+1)]*\
											tempVal_upper / (tempVal_upper-tempVal_lower);
			  }
				   new_equation_upper[inputSize+i*(inputSize+1)] -= tempVal_upper*tempVal_lower/\
															 (tempVal_upper-tempVal_lower);

						 if(-tempVal_lower>tempVal_upper)
						{
							 for(k=0;k<inputSize+1;k++){
							   new_equation_lower[k+i*(inputSize+1)] = 0;
							  }
						 }
						 states[layer+1][i] = 1;
				}
				if(RELAX_MOD==2)
				{
					if(upper_s_lower<0.0){
						for(k=0;k<inputSize+1;k++){
							new_equation_upper[k+i*(inputSize+1)] =\
													new_equation_upper[k+i*(inputSize+1)]*\
													tempVal_upper / (tempVal_upper-upper_s_lower);
						}
						new_equation_upper[inputSize+i*(inputSize+1)] -= tempVal_upper*upper_s_lower/\
															(tempVal_upper-upper_s_lower);
					}


					if(lower_s_upper<0.0){
						for(k=0;k<inputSize+1;k++){
							new_equation_lower[k+i*(inputSize+1)] = 0;
						}
					}
				   else{
						  for(k=0;k<inputSize+1;k++){
							new_equation_lower[k+i*(inputSize+1)] =\
													new_equation_lower[k+i*(inputSize+1)]*\
													lower_s_upper / (lower_s_upper- tempVal_lower);
						}
					}

					states[layer+1][i] = 1;
				}
			}
			}
		}
            else{

            tempVal_upper = tempVal_lower = 0.0;
            lower_s_upper = upper_s_lower = 0.0;

            if(NEED_OUTWARD_ROUND){
                for(k=0;k<inputSize;k++){
                    if(new_equation_lower[k+i*(inputSize+1)]>=0){
                        tempVal_lower += new_equation_lower[k+i*(inputSize+1)] * input->lower_matrix.data[k]-OUTWARD_ROUND;
                        lower_s_upper += new_equation_lower[k+i*(inputSize+1)] * input->upper_matrix.data[k]-OUTWARD_ROUND;
                    }
                    else{
                        tempVal_lower += new_equation_lower[k+i*(inputSize+1)] * input->upper_matrix.data[k]-OUTWARD_ROUND;
                        lower_s_upper += new_equation_lower[k+i*(inputSize+1)] * input->lower_matrix.data[k]-OUTWARD_ROUND;
                    }
                    if(new_equation_upper[k+i*(inputSize+1)]>=0){
                        tempVal_upper += new_equation_upper[k+i*(inputSize+1)] * input->upper_matrix.data[k]+OUTWARD_ROUND;
                        upper_s_lower += new_equation_upper[k+i*(inputSize+1)] * input->lower_matrix.data[k]+OUTWARD_ROUND;
                    }
                    else{
                        tempVal_upper += new_equation_upper[k+i*(inputSize+1)] * input->lower_matrix.data[k]+OUTWARD_ROUND;
                        upper_s_lower += new_equation_upper[k+i*(inputSize+1)] * input->upper_matrix.data[k]+OUTWARD_ROUND;
                    }
                }
            }
            else{
                for(k=0;k<inputSize;k++){
                    if(layer==0){
                        if(new_equation_lower[k+i*(inputSize+1)]!=new_equation_upper[k+i*(inputSize+1)]){
                            printf("wrong!\n");
                        }
                    }
                    if(new_equation_lower[k+i*(inputSize+1)]>=0){
                        tempVal_lower = tempVal_lower + new_equation_lower[k+i*(inputSize+1)] * input->lower_matrix.data[k];
                        lower_s_upper =lower_s_upper+ new_equation_lower[k+i*(inputSize+1)] * input->upper_matrix.data[k];
                    }
                    else{
                        tempVal_lower = tempVal_lower + new_equation_lower[k+i*(inputSize+1)] * input->upper_matrix.data[k];
                        lower_s_upper =lower_s_upper+ new_equation_lower[k+i*(inputSize+1)] * input->lower_matrix.data[k];
                    }
                    if(new_equation_upper[k+i*(inputSize+1)]>=0){
                        tempVal_upper = tempVal_upper + new_equation_upper[k+i*(inputSize+1)] * input->upper_matrix.data[k];
                        upper_s_lower = upper_s_lower + new_equation_upper[k+i*(inputSize+1)] * input->lower_matrix.data[k];
                    }
                    else{
                        tempVal_upper = tempVal_upper + new_equation_upper[k+i*(inputSize+1)] * input->lower_matrix.data[k];
                        upper_s_lower = upper_s_lower + new_equation_upper[k+i*(inputSize+1)] * input->upper_matrix.data[k];
                    }

                }
            }
            tempVal_lower =tempVal_lower + new_equation_lower[inputSize+i*(inputSize+1)];
            lower_s_upper =lower_s_upper + new_equation_lower[inputSize+i*(inputSize+1)];
            tempVal_upper = tempVal_upper+new_equation_upper[inputSize+i*(inputSize+1)];
            upper_s_lower = upper_s_lower + new_equation_upper[inputSize+i*(inputSize+1)];


			if(is_first_time==0)
			{
				if(value_lower[layer+1][i]>tempVal_lower)
					tempVal_lower=value_lower[layer+1][i];
				else
				    value_lower[layer+1][i]=tempVal_lower;
				if(value_upper[layer+1][i]>tempVal_upper)
				    value_upper[layer+1][i]=tempVal_upper;
				else
					tempVal_upper=value_upper[layer+1][i];
			}
			else
			{
				value_lower[layer+1][i]=tempVal_lower;
				value_upper[layer+1][i]=tempVal_upper;
			}



			output->upper_matrix.data[i] = tempVal_upper;
			output->lower_matrix.data[i] = tempVal_lower;
            }
        }

        memcpy(equation_upper, new_equation_upper, sizeof(float)*(inputSize+1)*maxLayerSize);
        memcpy(equation_lower, new_equation_lower, sizeof(float)*(inputSize+1)*maxLayerSize);
        equation_inteval.lower_matrix.row = equation_inteval.upper_matrix.row =\
                                                         new_equation_inteval.lower_matrix.row;
        equation_inteval.lower_matrix.col = equation_inteval.upper_matrix.col =\
                                                         new_equation_inteval.lower_matrix.col;
    }
    return 1;
}

int forward_prop(struct NNet *network, struct Matrix *input, struct Matrix *output)
{

    int i,j,layer;

    struct NNet* nnet = network;
    int numLayers    = nnet->numLayers;
    int inputSize    = nnet->inputSize;
    int outputSize   = nnet->outputSize;

    float z[nnet->maxLayerSize];
    float a[nnet->maxLayerSize];
    struct Matrix Z = {z, 1, inputSize};
    struct Matrix A = {a, 1, inputSize};

    memcpy(Z.data, input->data, nnet->inputSize*sizeof(float));

    for(layer=0;layer<numLayers;layer++){
        A.row = nnet->bias[layer].row;
        A.col = nnet->bias[layer].col;
        memcpy(A.data, nnet->bias[layer].data, A.row*A.col*sizeof(float));

        matmul_with_bias(&Z, &nnet->weights[layer], &A);
        if(layer<numLayers-1){
            relu(&A);
        }
        memcpy(Z.data, A.data, A.row*A.col*sizeof(float));
        Z.row = A.row;
        Z.col = A.col;
        
    }

    memcpy(output->data, A.data, A.row*A.col*sizeof(float));
    output->row = A.row;
    output->col = A.col;
//    printf("The output is: \n");
//    printMatrix(output);

    return 1;
}



float refine_by_backward_1(struct NNet *nnet,lprec *lp, float *equation, int start_place, int *rule_num, int inputSize, int is_max, float *refined_input_upper, float *refined_input_lower)
{
	//struct NNet* nnet = network;
    int unsat = 0;
    int Ncol = inputSize;
    REAL row[Ncol+1];
    int colno[Ncol+1];
    int ret;
    memset(row, 0, Ncol*sizeof(float));
    set_add_rowmode(lp, TRUE);
    for(int j=1;j<Ncol+1;j++){
        row[j] = equation[start_place+j-1];
    }
	add_constraintex(lp, 1, row,NULL, GE, -equation[inputSize+start_place]+0.50012);
    set_add_rowmode(lp, FALSE);
    return unsat;
}


float refine_by_backward_2(struct NNet *network,lprec *lp, float *equation, int start_place, int *rule_num, int inputSize, int is_max, float *refined_input_upper, float *refined_input_lower)
{
	struct NNet* nnet = network;
//	printf("run refine_by_backward_2\n");
	fflush(stdout);
    int unsat = 0;
    int Ncol = inputSize;
    REAL row[Ncol+1];
    int colno[Ncol+1];
    int ret;
    memset(row, 0, Ncol*sizeof(float));
    set_add_rowmode(lp, TRUE);
    for(int o=0;o<nnet->outputSize;o++)
    {
       if(o!=nnet->target)// && output_upper_lower[nnet->target]>0)
      {	
	    start_place=o*(inputSize+1);
	    for(int j=1;j<Ncol+1;j++){
		row[j] = equation[start_place+j-1];
	//	printf("row[%d] is %f",j,row[j]);
	    }
	    //只增加了一个约束，当性质取反是a&&b时则需要添加两个
	    if(is_max){
		add_constraintex(lp, 1, row,NULL, GE, -equation[inputSize+start_place]);
	    //    set_maxim(lp);
	    }
	    else{
	    	 add_constraintex(lp, 1, row,NULL, LE, -equation[inputSize+start_place]);
	    //    set_minim(lp);
	    }
		fflush(stdout);
	    *rule_num =  *rule_num+1;
   	}
     }
 set_add_rowmode(lp, FALSE);
 return unsat;
}


int refine_by_backward(struct NNet *network, struct Interval *input, \
		     float *output_symbl_upper, float *output_symbl_lower, lprec *lp, int *rule_num, int property)
{
   int unsat=0;
   struct NNet* nnet = network;
   int inputSize= nnet->inputSize;
   int layerSize = nnet->layerSizes;
   int target=nnet->target;
   if(property==1) //
	//   if(output_upper_lower[target]<0.5011)
	     unsat = refine_by_backward_1(nnet,lp, output_symbl_upper, target*(inputSize+1), rule_num, inputSize, 1, input->upper_matrix.data,input->lower_matrix.data);
   if(property==2) //
 	//   if(output_upper_lower[target]<0.5011)
 	     unsat = refine_by_backward_2(nnet,lp,output_symbl_upper, target*(inputSize+1), rule_num, inputSize, 1, input->upper_matrix.data,input->lower_matrix.data);
   if(property==3)
	   unsat = refine_by_backward_2(nnet,lp,output_symbl_lower, target*(inputSize+1), rule_num, inputSize, 0, input->upper_matrix.data,input->lower_matrix.data);
   if(property==4)
 	   unsat = refine_by_backward_2(nnet,lp,output_symbl_lower, target*(inputSize+1), rule_num, inputSize, 0, input->upper_matrix.data,input->lower_matrix.data);

   return unsat;
}


int check_property_by_forward_backward_refining(
		struct NNet *nnet, struct Interval *input,
//		 struct Interval *output,
		 float **value_upper, float **value_lower,
		 float **symbol_upper, float **symbol_lower, int **states,
		 float *equation_upper, float *equation_lower,
		 float *new_equation_upper, float *new_equation_lower,
		 lprec *lp, int *rule_num,int property,int need_prop,int depth,int split_layer_pre,int change_split_layer,int input_refined)
{
	int inputSize=nnet->inputSize;
	int numLayers=nnet->numLayers;
	int maxLayerSize=nnet->maxLayerSize;
    int verify_flag=0;
		if(need_prop)
		{
		    float o_upper1[nnet->outputSize], o_lower1[nnet->outputSize];
		    struct Interval output = {
		            (struct Matrix){o_lower1, nnet->outputSize, 1},
		            (struct Matrix){o_upper1, nnet->outputSize, 1}
		        };
			forward_prop_value_symbl_linear2(nnet, input, &output,\
			                                      value_upper,value_lower,//output_lower_upper,output_upper_lower,
			                                     symbol_upper,symbol_lower,states,\
			                                     equation_upper, equation_lower, new_equation_upper,new_equation_lower,\
			                                     depth,0,split_layer_pre,change_split_layer,input_refined);

			verify_flag = check_functions_wit_constraints(nnet, &output,symbol_upper,symbol_lower,lp);
			if(verify_flag==1) //property is verified to be true!
			{
			    	return verify_flag;
			}
			    verify_flag = refine_by_backward(nnet,input, symbol_upper[numLayers],symbol_lower[numLayers],lp,rule_num,property);
		}

			int split_layer=0,split_node=0,need_split=0;

			//split_by_dependency
			if(SPLITING_METHOD==2)
				need_split=choose_spliting_node2(nnet,value_upper,value_lower,states,&split_layer,&split_node);
//			else if(SPLITING_METHOD==1)
//				need_split=choose_spliting_node(nnet,value_upper_pre,value_lower_pre,value_upper,value_lower,states,0,&split_layer,&split_node);
		    else if(SPLITING_METHOD==0)
		    	need_split=choose_spliting_node_by_dependency(nnet,input,value_upper, value_lower,symbol_upper,symbol_lower,states, &split_layer, &split_node);
			else if(SPLITING_METHOD==1)
				need_split=choose_spliting_node3(nnet,value_upper,value_lower,states,0,&split_layer,&split_node);
			else if(SPLITING_METHOD==3)
				need_split=choose_spliting_node4(nnet,value_upper,value_lower,states,0,&split_layer,&split_node);
			else if(SPLITING_METHOD==4)//non_ful, random
				need_split=choose_spliting_node5(nnet,value_upper,value_lower,states,0,&split_layer,&split_node);
	    	if(need_split==0)
			{
	    	 return 1;
			}
            else
            {
            	if(split_layer>split_layer_pre)
            	{
            		change_split_layer=1;
            	}
            	else
            		change_split_layer=0;

            	verify_flag=split_by_predicates(nnet,input,value_upper,value_lower,symbol_upper,symbol_lower,states,
            			equation_upper, equation_lower, new_equation_upper,new_equation_lower,
            	    		lp,rule_num,property,split_layer,split_node,depth,change_split_layer);
            }
    return verify_flag;
}
