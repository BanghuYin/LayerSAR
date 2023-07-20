#include "nnet.h"
#include "split.h"


int PROPERTY = 0;
float INF = 1;
float Timeout=600.0;

struct timeval start,finish,last_finish;

struct NNet *load_conv_network(const char* filename, int img)
{
    //Load file and check if it exists
    FILE *fstream = fopen(filename,"r");

    if (fstream == NULL)
    {
        printf("Wrong network!\n");
        exit(1);
    }
    //Initialize variables
    int bufferSize = 650000;
    char *buffer = (char*)malloc(sizeof(char)*bufferSize);
    char *record, *line;
    int i=0, layer=0, row=0, j=0, param=0;

    struct NNet *nnet = (struct NNet*)malloc(sizeof(struct NNet));

    //memset(nnet, 0, sizeof(struct NNet));
    //Read int parameters of neural network

    line=fgets(buffer,bufferSize,fstream);
    while (strstr(line, "//")!=NULL)
        line=fgets(buffer,bufferSize,fstream); //skip header lines
    record = strtok(line,",\n");
    nnet->numLayers = atoi(record);
    nnet->inputSize = atoi(strtok(NULL,",\n"));
    nnet->outputSize = atoi(strtok(NULL,",\n"));
    nnet->maxLayerSize = atoi(strtok(NULL,",\n"));

    //Allocate space for and read values of the array members of the network
    nnet->layerSizes = (int*)malloc(sizeof(int)*(nnet->numLayers+1));
    line = fgets(buffer,bufferSize,fstream);
    record = strtok(line,",\n");
    for (i = 0; i<((nnet->numLayers)+1); i++)
    {
        nnet->layerSizes[i] = atoi(record);
        record = strtok(NULL,",\n");
    }
    //Load the symmetric paramter
    nnet->symmetric = 0;
    //Load Min and Max values of inputs
    nnet->min = MIN_PIXEL;
    nnet->max = MAX_PIXEL;
    //Load Mean and Range of inputs
    nnet->mean = MIN_PIXEL;
    nnet->range = MAX_PIXEL;
    nnet->layerTypes = (int*)malloc(sizeof(int)*nnet->numLayers);
    nnet->convLayersNum = 0;
    line = fgets(buffer,bufferSize,fstream);
    record = strtok(line,",\n");
    for (i = 0; i<nnet->numLayers; i++)
    {
        nnet->layerTypes[i] = atoi(record);
        if(nnet->layerTypes[i]==1){
            nnet->convLayersNum++;
        }
        record = strtok(NULL,",\n");
    }
    //initial convlayer parameters
    nnet->convLayer = (int**)malloc(sizeof(int *)*nnet->convLayersNum);
    for(i=0; i<nnet->convLayersNum;i++){
        nnet->convLayer[i] = (int*)malloc(sizeof(int)*5);
    }

    for(int cl=0;cl<nnet->convLayersNum;cl++){
        line = fgets(buffer,bufferSize,fstream);
        record = strtok(line,",\n");
        for (i = 0; i<5; i++){
            nnet->convLayer[cl][i] = atoi(record);
            //printf("%d,", nnet->convLayer[cl][i]);
            record = strtok(NULL,",\n");
        }
        //printf("\n");
    }

    //Allocate space for matrix of Neural Network
    //
    //The first dimension will be the layer number
    //The second dimension will be 0 for weights, 1 for biases
    //The third dimension will be the number of neurons in that layer
    //The fourth dimension will be the number of inputs to that layer
    //
    //Note that the bias array will have only number per neuron, so
    //    its fourth dimension will always be one
    //
    nnet->matrix = (float****)malloc(sizeof(float *)*(nnet->numLayers));
    for (layer = 0; layer<nnet->numLayers; layer++){
        if(nnet->layerTypes[layer]==0){
            nnet->matrix[layer] = (float***)malloc(sizeof(float *)*2);
            nnet->matrix[layer][0] = (float**)malloc(sizeof(float *)*nnet->layerSizes[layer+1]);
            nnet->matrix[layer][1] = (float**)malloc(sizeof(float *)*nnet->layerSizes[layer+1]);
            for (row = 0; row<nnet->layerSizes[layer+1]; row++){
                nnet->matrix[layer][0][row] = (float*)malloc(sizeof(float)*nnet->layerSizes[layer]);
                nnet->matrix[layer][1][row] = (float*)malloc(sizeof(float));
            }
        }
    }

    nnet->conv_matrix = (float****)malloc(sizeof(float *)*nnet->convLayersNum);
    for(layer=0;layer<nnet->convLayersNum;layer++){
        int out_channel = nnet->convLayer[layer][0];
        int in_channel = nnet->convLayer[layer][1];
        int kernel_size = nnet->convLayer[layer][2]*nnet->convLayer[layer][2];
        nnet->conv_matrix[layer]=(float***)malloc(sizeof(float*)*out_channel);
        for(int oc=0;oc<out_channel;oc++){
            nnet->conv_matrix[layer][oc] = (float**)malloc(sizeof(float*)*in_channel);
            for(int ic=0;ic<in_channel;ic++){
                nnet->conv_matrix[layer][oc][ic] = (float*)malloc(sizeof(float)*kernel_size);
            }

        }
    }

    nnet->conv_bias = (float**)malloc(sizeof(float*)*nnet->convLayersNum);
    for(layer=0;layer<nnet->convLayersNum;layer++){
        int out_channel = nnet->convLayer[layer][0];
        nnet->conv_bias[layer] = (float*)malloc(sizeof(float)*out_channel);
    }

    layer = 0;
    param = 0;
    i=0;
    j=0;
    char *tmpptr=NULL;

    int oc=0, ic=0, kernel=0;
    int out_channel=0,in_channel=0,kernel_size=0;

    //Read in parameters and put them in the matrix
    float w = 0.0;
    while((line=fgets(buffer,bufferSize,fstream))!=NULL){
        if(nnet->layerTypes[layer]==1){
            out_channel = nnet->convLayer[layer][0];
            in_channel = nnet->convLayer[layer][1];
            kernel_size = nnet->convLayer[layer][2]*nnet->convLayer[layer][2];
            if(oc>=out_channel){
                if (param==0)
                {
                    param = 1;
                }
                else
                {
                    param = 0;
                    layer++;
                    if(nnet->layerTypes[layer]==1){
                        out_channel = nnet->convLayer[layer][0];
                        in_channel = nnet->convLayer[layer][1];
                        kernel_size = nnet->convLayer[layer][2]*nnet->convLayer[layer][2];
                    }
                }
                oc=0;
                ic=0;
                kernel=0;
            }
        }
        else{
            if(i>=nnet->layerSizes[layer+1]){
                if (param==0)
                {
                    param = 1;
                }
                else
                {
                    param = 0;
                    layer++;
                }
                i=0;
                j=0;
            }
        }

        if(nnet->layerTypes[layer]==1){
            if(param==0){
                record = strtok_r(line,",\n", &tmpptr);
                while(record != NULL)
                {

                    w = (float)atof(record);
                    nnet->conv_matrix[layer][oc][ic][kernel] = w;
                    kernel++;
                    if(kernel==kernel_size){
                        kernel = 0;
                        ic++;
                    }
                    record = strtok_r(NULL, ",\n", &tmpptr);
                }
                tmpptr=NULL;
                kernel=0;
                ic=0;
                oc++;
            }
            else{
                record = strtok_r(line,",\n", &tmpptr);
                while(record != NULL)
                {

                    w = (float)atof(record);
                    nnet->conv_bias[layer][oc] = w;
                    record = strtok_r(NULL, ",\n", &tmpptr);
                }
                tmpptr=NULL;
                oc++;
            }
        }
        else{
            record = strtok_r(line,",\n", &tmpptr);
            while(record != NULL)
            {
                w = (float)atof(record);
                nnet->matrix[layer][param][i][j] = w;
                j++;
                record = strtok_r(NULL, ",\n", &tmpptr);
            }
            tmpptr=NULL;
            j=0;
            i++;
        }
    }
    //printf("load matrix done\n");

    float input_prev[nnet->inputSize];
    struct Matrix input_prev_matrix = {input_prev, 1, nnet->inputSize};
    float o[nnet->outputSize];
    struct Matrix output = {o, nnet->outputSize, 1};
    //printf("start load inputs\n");
    load_inputs(img, nnet->inputSize, input_prev);
    //printf("load inputs done\n");
    if(PROPERTY<500){
        normalize_input(nnet, &input_prev_matrix);
    }
    //printf("normalize_input done\n");
    evaluate_conv(nnet, &input_prev_matrix, &output);
 //   printMatrix(&output);

    float largest = -100000.0;
    for(int o=0;o<nnet->outputSize;o++){
        if(output.data[o]>largest){
            largest = output.data[o];
            nnet->target = o;
        }
    }
    float orig_weights[nnet->layerSizes[layer]];
    float orig_bias;
    struct Matrix *weights = malloc(nnet->numLayers*sizeof(struct Matrix));
    struct Matrix *bias = malloc(nnet->numLayers*sizeof(struct Matrix));

    for(int layer=0;layer<nnet->numLayers;layer++){
        if(nnet->layerTypes[layer]==1) continue;
        weights[layer].row = nnet->layerSizes[layer];
        weights[layer].col = nnet->layerSizes[layer+1];
        weights[layer].data =\
                    (float*)malloc(sizeof(float)*weights[layer].row * weights[layer].col);
        int n=0;
        if(0){
            /*
             * Make the weights of last layer to minus the target one.
             */
            if(layer==nnet->numLayers-1){
                orig_bias = nnet->matrix[layer][1][nnet->target][0];
                memcpy(orig_weights, nnet->matrix[layer][0][nnet->target],\
                                     sizeof(float)*nnet->layerSizes[layer]);
                for(int i=0;i<weights[layer].col;i++){
                    for(int j=0;j<weights[layer].row;j++){
                        weights[layer].data[n] = nnet->matrix[layer][0][i][j]-orig_weights[j];
                        n++;
                    }
                }
                bias[layer].col = nnet->layerSizes[layer+1];
                bias[layer].row = (float)1;
                bias[layer].data = (float*)malloc(sizeof(float)*bias[layer].col);
                for(int i=0;i<bias[layer].col;i++){
                    bias[layer].data[i] = nnet->matrix[layer][1][i][0]-orig_bias;
                }
            }
            else{
                for(int i=0;i<weights[layer].col;i++){
                    for(int j=0;j<weights[layer].row;j++){
                        weights[layer].data[n] = nnet->matrix[layer][0][i][j];
                        n++;
                    }
                }
                bias[layer].col = nnet->layerSizes[layer+1];
                bias[layer].row = (float)1;
                bias[layer].data = (float*)malloc(sizeof(float)*bias[layer].col);
                for(int i=0;i<bias[layer].col;i++){
                    bias[layer].data[i] = nnet->matrix[layer][1][i][0];
                }
            }
        }
        else{
            for(int i=0;i<weights[layer].col;i++){
                for(int j=0;j<weights[layer].row;j++){
                    weights[layer].data[n] = nnet->matrix[layer][0][i][j];
                    n++;
                }
            }
            bias[layer].col = nnet->layerSizes[layer+1];
            bias[layer].row = (float)1;
            bias[layer].data = (float*)malloc(sizeof(float)*bias[layer].col);
            for(int i=0;i<bias[layer].col;i++){
                bias[layer].data[i] = nnet->matrix[layer][1][i][0];
            }
        }

    }
    nnet->weights = weights;
    nnet->bias = bias;

    free(buffer);
    fclose(fstream);
    return nnet;
}

void destroy_conv_network(struct NNet *nnet)
{
    int i=0, row=0;
    if (nnet!=NULL)
    {
        for(i=0; i<nnet->numLayers; i++)
        {
            if(nnet->layerTypes[i]==1) continue;
            for(row=0;row<nnet->layerSizes[i+1];row++)
            {
                //free weight and bias arrays
                free(nnet->matrix[i][0][row]);
                free(nnet->matrix[i][1][row]);
            }
            //free pointer to weights and biases
            free(nnet->matrix[i][0]);
            free(nnet->matrix[i][1]);
            free(nnet->weights[i].data);
            free(nnet->bias[i].data);
            free(nnet->matrix[i]);
        }
        for(i=0;i<nnet->convLayersNum;i++){
            int kernel_size = nnet->convLayer[i][2]*nnet->convLayer[i][2];
            int in_channel = nnet->convLayer[i][1];
            int out_channel = nnet->convLayer[i][0];
            for(int oc=0;oc<out_channel;oc++){
                for(int ic=0;ic<in_channel;ic++){
                    free(nnet->conv_matrix[i][oc][ic]);
                }
                free(nnet->conv_matrix[i][oc]);
            }
            free(nnet->conv_matrix[i]);
            free(nnet->conv_bias[i]);
        }
        free(nnet->conv_bias);
        free(nnet->conv_matrix);
        for(i=0;i<nnet->convLayersNum;i++){
            free(nnet->convLayer[i]);
        }
        free(nnet->convLayer);
        free(nnet->weights);
        free(nnet->bias);
        free(nnet->layerSizes);
        free(nnet->layerTypes);
        free(nnet->matrix);
        free(nnet);
    }
}




void set_input_constraints(struct Interval *input, lprec *lp, int *rule_num){
    int Ncol = 784;
    REAL row[Ncol+1];
    int colno[Ncol+1];
    memset(row, 0, Ncol*sizeof(float));
    set_add_rowmode(lp, TRUE);
    for(int var=1;var<Ncol+1;var++){
        memset(colno, 0, Ncol*sizeof(int));

        colno[0] = var;
        row[0] = 1;
        add_constraintex(lp, 1, row, colno, LE, input->upper_matrix.data[var-1]);
        add_constraintex(lp, 1, row, colno, GE, input->lower_matrix.data[var-1]);
        *rule_num += 2;
    }
    set_add_rowmode(lp, FALSE);
}

float set_output_constraints(lprec *lp, float *equation, int start_place, int *rule_num, int inputSize, int is_max, float *output, float *input_prev){

    float time_spent=0.0;
    int unsat = 0;
    int Ncol = inputSize;
    REAL row[Ncol+1];
    int colno[Ncol+1];
    memset(row, 0, Ncol*sizeof(float));
    set_add_rowmode(lp, TRUE);
    for(int j=1;j<Ncol+1;j++){
        row[j] = equation[start_place+j-1];
    }
    if(is_max){
        //printf("greater than : %f\n",-equation[inputSize+start_place] );
        add_constraintex(lp, 1, row, NULL, GE, -equation[inputSize+start_place]);
        set_maxim(lp);
    }
    else{
        add_constraintex(lp, 1, row, NULL, LE, -equation[inputSize+start_place]);
        set_minim(lp);
    }

    set_add_rowmode(lp, FALSE);

    set_obj_fnex(lp, Ncol+1, row, NULL);
    *rule_num += 1;
    //write_lp(lp, "model3.lp");
    int ret = 0;

    //printf("in1\n");

    ret = solve(lp);

    //printf("in2,%d\n",ret);

    if(ret == OPTIMAL){
        int Ncol = inputSize;
        double row[Ncol+1];
        *output = get_objective(lp)+equation[inputSize+start_place];
        get_variables(lp, row);
        for(int j=0;j<Ncol;j++){
            input_prev[j] = (float)row[j];
        }
    }
    else{
        //printf("unsat!\n");
        unsat = 1;
    }


    return unsat;
}

void initialize_input_interval(struct NNet* nnet, int img, int inputSize, float *input, float *u, float *l){
    load_inputs(img, inputSize, input);
	for(int i =0;i<inputSize;i++){
		u[i] = input[i]+INF;
		l[i] = input[i]-INF;
	}
}


void load_inputs(int img, int inputSize, float *input){
    if(PROPERTY<500){
        if(img>=100000){
            printf("image over 100000!\n");
            exit(1);
        }
        char str[12];
        char image_name[100];

        if(PROPERTY==11) {
            char tmp[100] = "fmnist/image";
            strcpy(image_name, tmp);
        }
        else {
            char tmp[100] = "images/image";
            strcpy(image_name, tmp);
        }

        sprintf(str, "%d", img);
        FILE *fstream = fopen(strcat(image_name,str),"r");
        if (fstream == NULL)
        {
            printf("no input:%s!\n", image_name);
            exit(1);
        }
        int bufferSize = 10240*5;
        char *buffer = (char*)malloc(sizeof(char)*bufferSize);
        char *record, *line;
        line = fgets(buffer,bufferSize,fstream);
        record = strtok(line,",\n");
        for (int i = 0; i<inputSize; i++)
        {
            input[i] = atof(record);
            record = strtok(NULL,",\n");
        }
        free(buffer);
        fclose(fstream);
    }
    else{
        if(img>=100000){
            printf("image over 100000!\n");
            exit(1);
        }
        char str[12];
        char image_name[18] = "cars/image";
        sprintf(str, "%d", img);
        FILE *fstream = fopen(strcat(image_name,str),"r");
        if (fstream == NULL)
        {
            printf("no input:%s!\n", image_name);
            exit(1);
        }
        int bufferSize = 300000;
        char *buffer = (char*)malloc(sizeof(char)*bufferSize);
        char *record, *line;
        line = fgets(buffer,bufferSize,fstream);
        record = strtok(line,",\n");
        for (int i = 0; i<inputSize; i++)
        {

            input[i] = atof(record);
            record = strtok(NULL,",\n");
        }
        free(buffer);
        fclose(fstream);
    }
}


void denormalize_input(struct NNet *nnet, struct Matrix *input){
    for (int i=0; i<nnet->inputSize;i++)
    {
        input->data[i] = input->data[i]*(nnet->range) + nnet->mean;
    }
}

void normalize_input(struct NNet *nnet, struct Matrix *input){
    for (int i=0; i<nnet->inputSize;i++)
    {

        if (input->data[i]>nnet->max)
        {
            input->data[i] = (nnet->max-nnet->mean)/(nnet->range);
        }
        else if (input->data[i]<nnet->min)
        {
            input->data[i] = (nnet->min-nnet->mean)/(nnet->range);
        }
        else
        {
            input->data[i] = (input->data[i]-nnet->mean)/(nnet->range);
        }

        //input->data[i] = (input->data[i]-nnet->mean)/(nnet->range);
    }
}


void normalize_input_interval(struct NNet *nnet, struct Interval *input){
    normalize_input(nnet, &input->upper_matrix);
    normalize_input(nnet, &input->lower_matrix);
}


int forward_prop_conv(struct NNet *network, struct Matrix *input, struct Matrix *output){
    evaluate_conv(network, input, output);
    float t = output->data[network->target];
    for(int o=0;o<network->outputSize;o++){
        output->data[o] -= t;
    }
}



int evaluate_conv(struct NNet *network, struct Matrix *input, struct Matrix *output){
    int i,j,layer;

    struct NNet* nnet = network;
    int numLayers    = nnet->numLayers;
    int inputSize    = nnet->inputSize;
    int outputSize   = nnet->outputSize;

    float ****matrix = nnet->matrix;
    float ****conv_matrix = nnet->conv_matrix;

    float tempVal;
    float z[nnet->maxLayerSize];
    float a[nnet->maxLayerSize];


    //printf("start evaluate\n");
    for (i=0; i < nnet->inputSize; i++)
    {
        z[i] = input->data[i];
    }

    int out_channel=0, in_channel=0, kernel_size=0;
    int stride=0, padding=0;

    for (layer = 0; layer<(numLayers); layer++)
    {

        memset(a, 0, sizeof(float)*nnet->maxLayerSize);

        //printf("layer:%d %d\n",layer, nnet->layerTypes[layer]);
        if(nnet->layerTypes[layer]==0){
            for (i=0; i < nnet->layerSizes[layer+1]; i++){
                float **weights = matrix[layer][0];
                float **biases  = matrix[layer][1];
                tempVal = 0.0;

                //Perform weighted summation of inputs
                for (j=0; j<nnet->layerSizes[layer]; j++){
                    tempVal += z[j]*weights[i][j];
                }

                //Add bias to weighted sum
                tempVal += biases[i][0];

                //Perform ReLU
                if (tempVal<0.0 && layer<(numLayers-1)){
                    // printf( "doing RELU on layer %u\n", layer );
                    tempVal = 0.0;
                }
                a[i]=tempVal;
            }
            for(j=0;j<nnet->maxLayerSize;j++){
                //if(layer==2 && j<100) printf("%d %f\n",j, a[j]);
                //if(layer==5 && j<100) printf("%d %f\n",j, a[j]);
                z[j] = a[j];
            }
        }
        else{
            out_channel = nnet->convLayer[layer][0];
            in_channel = nnet->convLayer[layer][1];
            kernel_size = nnet->convLayer[layer][2];
            stride = nnet->convLayer[layer][3];
            padding = nnet->convLayer[layer][4];
            //size is the input size in each channel
            int size = sqrt(nnet->layerSizes[layer]/in_channel);
            //padding size is the input size after padding
            int padding_size = size+2*padding;
            //this is only for compressed model
            if(kernel_size%2==1){
                padding_size += 1;
            }
            //out_size is the output size in each channel after kernel
            int out_size = 0;
		float tmp_out_size =  (padding_size-(kernel_size-1)-1)/stride+1;
		if(tmp_out_size == (int)tmp_out_size){
			out_size = (int)tmp_out_size;
		}
		else{
			out_size = (int)(tmp_out_size)-1;
		}

            float *z_new = (float*)malloc(sizeof(float)*padding_size*padding_size*in_channel);
            memset(z_new, 0, sizeof(float)*padding_size*padding_size*in_channel);
            for(int ic=0;ic<in_channel;ic++){
                for(int h=0;h<size;h++){
                    for(int w=0;w<size;w++){
                        z_new[ic*padding_size*padding_size+padding_size*(h+padding)+w+padding] =\
                                                            z[ic*size*size+size*h+w];
                    }
                }
            }

            for(int oc=0;oc<out_channel;oc++){
                for(int oh=0;oh<out_size;oh++){
                    for(int ow=0;ow<out_size;ow++){
                        int start = ow*stride+oh*stride*padding_size;
                        for(int kh=0;kh<kernel_size;kh++){
                            for(int kw=0;kw<kernel_size;kw++){
                                for(int ic=0;ic<in_channel;ic++){
                                    a[oc*out_size*out_size+oh*out_size+ow] +=\
                                    conv_matrix[layer][oc][ic][kh*kernel_size+kw]*\
                                    z_new[ic*padding_size*padding_size+padding_size*kh+kw+start];
                                }
                            }
                        }
                        a[oc*out_size*out_size+ow+oh*out_size]+=nnet->conv_bias[layer][oc];
                    }
                }
            }
            for(j=0;j<nnet->maxLayerSize;j++){

                if(a[j]<0){
                    a[j] = 0;
                }
                z[j] = a[j];

            }
            free(z_new);
        }
    }

    for (i=0; i<outputSize; i++){
        output->data[i] = a[i];
    }
    
    return 1;
}


int forward_prop_value_symbl_linear2(struct NNet *network, struct Interval *input,
                                     struct Interval *output,
                                     float **value_upper, float **value_lower,
                                     float *output_lower_upper, float *output_upper_lower,
                                     float **symbol_upper, float **symbol_lower, int **states,
                                     int depth,int is_first_time,int split_layer_pre,int change_split_layer,int input_refined,int *output_map,lprec *lp,int *rule_num)
{

    pthread_mutex_lock(&lock);
    forward_count++;
    pthread_mutex_unlock(&lock);
    int i,j,k,layer;
    int node_cnt=0;
    if(DEBUG)
    {
    	printf("input ranges:\n");
    	printMatrix(&input->upper_matrix);
    	printMatrix(&input->lower_matrix);
    	fflush(stdout);
    }

    struct NNet* nnet = network;

    int numLayers    = nnet->numLayers;
    int inputSize    = nnet->inputSize;
    int outputSize   = nnet->outputSize;
    int maxLayerSize = nnet->maxLayerSize;

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
        symbol_lower[0][i*(inputSize+1)+i] = 1;
        symbol_upper[0][i*(inputSize+1)+i] = 1;
        value_lower[0][i] = input->lower_matrix.data[i];
        value_upper[0][i] = input->upper_matrix.data[i];
    }

    int begin_layer=0;

    for (layer = 0; layer<(numLayers); layer++)
    {
    	struct Matrix bias = nnet->bias[layer];

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
    	//                printf("p[i] is positive:%f\n", p[i]);
    	            }
    	            else{
    	                n[i] = weights.data[i];
    	//                printf("p[i] is negative:%f", n[i]);
    	            }
    	        }

        if(layer!=0) memset(new_equation_upper, 0, sizeof(float)*(inputSize+1)*maxLayerSize);
        if(layer!=0) memset(new_equation_lower, 0, sizeof(float)*(inputSize+1)*maxLayerSize);{
        if(DEBUG)
        {
        	printf("equation in this layer is:\n");
           printMatrix(&equation_inteval.upper_matrix);
           printMatrix(&equation_inteval.lower_matrix);
        }

        matmul(&equation_inteval.upper_matrix, &pos_weights, &new_equation_inteval.upper_matrix);
        matmul_with_bias(&equation_inteval.lower_matrix, &neg_weights, &new_equation_inteval.upper_matrix);

        matmul(&equation_inteval.lower_matrix, &pos_weights, &new_equation_inteval.lower_matrix);
        matmul_with_bias(&equation_inteval.upper_matrix, &neg_weights, &new_equation_inteval.lower_matrix);

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
                	//printf("not reach at first time\n");
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
                output_lower_upper[i]=lower_s_upper;
                output_upper_lower[i]=upper_s_lower;
                if(i!=nnet->target){
                                                  float upper_err=0, lower_err=0;
                                                  int temp=(inputSize+1)*inputSize;
                                                  float new_equation[temp];
                                                  for(int k=0;k<inputSize+1;k++){
                                                      new_equation[k+i*(inputSize+1)] = new_equation_upper[k+i*(inputSize+1)]-new_equation_lower[k+nnet->target*(inputSize+1)];
                                                  }


                                                  float upper = 0.0;
                                                  float input_prev[inputSize];
                                                  struct Matrix input_prev_matrix = {input_prev, 1, inputSize};
                                                  memset(input_prev, 0, sizeof(float)*inputSize);
                                                  float o[outputSize];
                                                  struct Matrix output_matrix = {o, outputSize, 1};
                                                  memset(o, 0, sizeof(float)*outputSize);
                                              	   lprec *lp_t;
                                              	   lp_t=copy_lp(lp);
                                              	 if(output_map[i]==1)
                                                  {
                                                      if(!set_output_constraints(lp_t, new_equation, i*(inputSize+1), rule_num, inputSize, MAX, &upper, input_prev)){
                                                      //    need_to_split = 1;

                                                          if(NEED_PRINT){
                                                              printf("%d--Objective value: %f\n", i, upper);
                                                          }
                                                          check_adv2(nnet, &input_prev_matrix);
                                                          if(adv_found){
                                                              return 0;
                                                          }
                                                          else if(0)
                                                          {
                                                        	  if(!set_output_constraints(lp_t, new_equation, i*(inputSize+1), rule_num, inputSize, MIN, &upper, input_prev)){
                                                        	                                                        //    need_to_split = 1;

                                                        	                                                            if(NEED_PRINT){
                                                        	                                                                printf("%d--Objective value: %f\n", i, upper);
                                                        	                                                            }
                                                        	                                                            check_adv2(nnet, &input_prev_matrix);
                                                        	                                                            if(adv_found){
                                                        	                                                                return 0;
                                                        	                                                            }
                                                        	  }
                                                          }
                                                      }
                                                      else{
                                                   	   output_map[i] = 0;
                                                          if(NEED_PRINT){
                                                              printf("%d--unsat\n", i);
                                                          }
                                                      }
                                                  }
                                                  delete_lp(lp_t);

                                              }


            }

            node_cnt++;

        }

        memcpy(equation_upper, new_equation_upper, sizeof(float)*(inputSize+1)*maxLayerSize);
        memcpy(equation_lower, new_equation_lower, sizeof(float)*(inputSize+1)*maxLayerSize);
        equation_inteval.lower_matrix.row = equation_inteval.upper_matrix.row =\
                                                         new_equation_inteval.lower_matrix.row;
        equation_inteval.lower_matrix.col = equation_inteval.upper_matrix.col =\
                                                         new_equation_inteval.lower_matrix.col;
    }


	free(equation_upper);
	free(equation_lower);
	free(new_equation_upper);
	free(new_equation_lower);
    return 1;
}

int find_counter_example(struct NNet *nnet, float **symbol_upper, float **symbol_lower,lprec *lp_o,int *output_map)
{
	lprec *lp_t;
	lp_t=copy_lp(lp_o);
	int inputSize=nnet->inputSize;
	int maxLayerSize=nnet->maxLayerSize;
	int numLayers=nnet->numLayers;
	int Ncol = inputSize;
	REAL row[Ncol+1];
	REAL row_total[Ncol+1];
	memset(row, 0, Ncol*sizeof(float));
	memset(row_total, 0, Ncol*sizeof(float));

	set_add_rowmode(lp_t, TRUE);
    for(int i=0;i<=nnet->outputSize;i++)
    {
    	if(output_map[i]==1)
    	{
			for(int j=1;j<Ncol+1;j++){
				row[j] = symbol_upper[nnet->numLayers][j+i*(inputSize+1)-1]-symbol_lower[nnet->numLayers][j+nnet->target*(inputSize+1)-1];
				row_total[j]=row_total[j]+row[j];
			}
		    add_constraint(lp_t, row, GE,-symbol_upper[nnet->numLayers][inputSize+i*(inputSize+1)]+symbol_lower[nnet->numLayers][inputSize+nnet->target*(inputSize+1)]);
    	}
    }
	set_add_rowmode(lp_t, FALSE);
    set_maxim(lp_t);
    set_obj_fnex(lp_t, Ncol+1, row_total, NULL);

      int ret = 0;


      ret = solve(lp_t);
		float input_prev[inputSize];
		struct Matrix input_prev_matrix = {input_prev, 1, inputSize};
		memset(input_prev, 0, sizeof(float)*inputSize);

      if(ret == OPTIMAL){
          int Ncol = inputSize;
          double row_r[Ncol+1];
          double temp= get_objective(lp_t);
          get_variables(lp_t, row_r);
          for(int j=0;j<Ncol;j++){
              input_prev[j] = (float)row_r[j];
          }
      	check_adv2(nnet, &input_prev_matrix);
      			if(adv_found){
      				delete_lp(lp_t);
      			return 1;
      			}
      			else
      			{
      				 set_minim(lp_t);
      				 ret = solve(lp_t);
      			    if(ret == OPTIMAL){
      			    //	  printf("reach here for ce\n");
      			          int Ncol = inputSize;
      			          double row_r[Ncol+1];
      			          double temp= get_objective(lp_t);
      			          get_variables(lp_t, row_r);
      			          for(int j=0;j<Ncol;j++){
      			              input_prev[j] = (float)row_r[j];
      			          }
      			      	check_adv2(nnet, &input_prev_matrix);
      			      			if(adv_found){
      			      				delete_lp(lp_t);
      			      			return 1;
      			      			}
      			    }
      			}
      }
      delete_lp(lp_t);
   return 0;
}

void refine_interval_values(struct NNet *nnet, float **symbol_upper, float **symbol_lower,lprec *lp, float **value_upper, float **value_lower,int **states)
{
	int inputSize=nnet->inputSize;
	int numLayers=nnet->numLayers;
	int maxLayerSize=nnet->maxLayerSize;
	int Ncol = inputSize;
	REAL row[Ncol+1];
	memset(row, 0, Ncol*sizeof(float));
    for (int layer = 1; layer<(numLayers); layer++)
    {
    	//printf("layer is %d, layersize is %d\n",layer,nnet->layerSizes[layer]);
    	for (int i=0; i < nnet->layerSizes[layer]; i++)
    	{
    		if(states[layer][i]==1)
			{
				for(int j=1;j<Ncol+1;j++)
					row[j] = symbol_upper[layer][j+i*(inputSize+1)-1];
				set_maxim(lp);
				set_obj_fnex(lp, Ncol+1, row, NULL);
				int ret = solve(lp);
				if(ret == OPTIMAL){
				  double temp= get_objective(lp);
				  double max_value=temp+symbol_upper[layer][inputSize+i*(inputSize+1)];
				  if(max_value<=0)
				  {
		//			  printf("reach here1\n");
					  states[layer][i]=0;
				  }
				  if(max_value<value_upper[layer][i])
					  value_upper[layer][i]=max_value;
				}
				for(int j=1;j<Ncol+1;j++)
					row[j] = symbol_lower[layer][j+i*(inputSize+1)-1];
				set_minim(lp);
				set_obj_fnex(lp, Ncol+1, row, NULL);
				ret = solve(lp);
				if(ret == OPTIMAL){
				  double temp= get_objective(lp);
				  double min_value=temp+symbol_lower[layer][inputSize+i*(inputSize+1)];
				  if(min_value>=0)
				  {
		//			  printf("reach here2\n");
					  states[layer][i]=2;
				  }
				  if(min_value>value_lower[layer][i])
					  value_lower[layer][i]=min_value;
				}
			}
    	}

    }
}

int check_property_by_forward_backward_refining(
		struct NNet *network, struct Interval *input,
		 struct Interval *output,
		 float **value_upper, float **value_lower,
		 float **symbol_upper, float **symbol_lower, int **states,
		 lprec *lp, int *rule_num,int property,int need_prop,int depth,int split_layer_pre,int change_split_layer,int input_refined,int *output_map)
{
	gettimeofday(&start1, NULL);
	double time_spent = ((float) (start1.tv_sec - start.tv_sec) * 1000000
			+ (float) (start1.tv_usec - start1.tv_usec)) / 1000000;
    if(time_spent>=Timeout)
    {
    }
    if(can_t_prove==1)
    	return 1;

	struct NNet* nnet=network;
	int inputSize=nnet->inputSize;
	int numLayers=nnet->numLayers;
	int maxLayerSize=nnet->maxLayerSize;
    int verify_flag=0;
    int cont_r1,cont_r2;

    if(DEBUG)
    {
		printf("\n The input ranges is:\n");
		printMatrix(&input->upper_matrix);
		printMatrix(&input->lower_matrix);
		printf("\n************************\n");
    }

		float output_lower_upper[nnet->outputSize], output_upper_lower[nnet->outputSize];

		if(need_prop==1)//&&input_refined==1)
		{
			forward_prop_value_symbl_linear2(nnet, input, output,\
			                                      value_upper,value_lower,output_lower_upper,output_upper_lower,\
			                                     symbol_upper,symbol_lower,states,\
			                                     depth,0,split_layer_pre,change_split_layer,input_refined,output_map,lp,rule_num);
				verify_flag = check_functions_norm(nnet, output);
		}
	    if(verify_flag==1) //property is verified to be true!
	    {
	    	return verify_flag;
	    }
	    verify_flag=1;
	    for(int i=0;i<nnet->outputSize;i++)
	    {
	         if(output_map[i]==1)
	         		verify_flag=0;

	    }
	    if(verify_flag==1) //property is verified to be true!
	    {
	    	return verify_flag;
	    }
	    int adv_get=find_counter_example(nnet,symbol_upper,symbol_lower,lp,output_map);
		if(adv_get)
			return 1;


		refine_interval_values(nnet, symbol_upper, symbol_lower,lp,value_upper,value_lower,states);
	    for(int i=0;i<nnet->outputSize;i++)
	    {
	         if(output_map[i]==1)
	         		verify_flag=0;
	    }
	    cont_r2=0;

		{

			int split_layer=0,split_node=0,need_split=0;


				need_split=choose_spliting_node2(nnet,value_upper,value_lower,states,0,&split_layer,&split_node);

			fflush(stdout);

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
            	fflush(stdout);

            	verify_flag=split_by_predicates(nnet,input,output,value_upper,value_lower,symbol_upper,symbol_lower,states,
            	    		lp,rule_num,property,split_layer,split_node,depth,change_split_layer,output_map);
            	if(verify_flag==1)
            	{
            		return verify_flag;
            	}
            	else
            	{
                        printf("Never reach here!\n");
            		return verify_flag;
            	}
            }
         }
    return verify_flag;
}
