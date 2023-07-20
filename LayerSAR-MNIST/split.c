#include "split.h"

#define MAX_THREAD 16
#define MIN_DEPTH_PER_THREAD 5 

int NEED_PRINT = 0;
int adv_found = 0;
int forward_count=0;
int count = 0;

int m_depth =0;

int CHECK_ADV_MODE = 0;

int DEBUG = 0;
int can_t_prove = 0;

struct timeval start,finish,last_finish,start1,finish1;


int check_not_max_norm(struct NNet *nnet, struct Interval *output){
    float t = output->lower_matrix.data[nnet->target];
	for(int i=0;i<nnet->outputSize;i++){
	    if(output->upper_matrix.data[i]>t && i != nnet->target){
			return 0;
		}
	}
	return 1;
}

int check_not_max1(struct NNet *nnet, struct Matrix *output){
    for(int i=0;i<nnet->outputSize;i++){
        if(output->data[i]>0 && i != nnet->target){
            return 1;
        }
    }
    return 0;
}

int check_functions_norm(struct NNet *nnet, struct Interval *output){
    return check_not_max_norm(nnet, output);
}

int check_functions_norm1(struct NNet *nnet, struct Matrix *output){
    return check_not_max1(nnet, output);
}


void check_adv2(struct NNet* nnet, struct Matrix *adv){
    float out[nnet->outputSize];
    struct Matrix output = {out, nnet->outputSize, 1};
    forward_prop_conv(nnet, adv, &output);
    int is_adv = 0;
    // printMatrix(&output);
    is_adv = check_functions_norm1(nnet, &output);
    if(is_adv){
        printf("adv found:\n");
        //printMatrix(adv);
        printMatrix(&output);
        int adv_output = 0;
        for(int i=0;i<nnet->outputSize;i++){
            if(output.data[i]>0 && i != nnet->target){
                    adv_output = i;
            }
        }
        printf("%d ---> %d\n", nnet->target, adv_output);
        pthread_mutex_lock(&lock);
        adv_found = 1;
        pthread_mutex_unlock(&lock);
    }
}


void check_adv1(struct NNet* nnet, struct Interval *input)
{

    float a[nnet->inputSize];
    struct Matrix adv = {a, 1, nnet->inputSize};

    for (int i=0;i<nnet->inputSize;i++) {
        float upper = input->upper_matrix.data[i];
        float lower = input->lower_matrix.data[i];
        float middle = (lower+upper)/2;

        a[i] = middle;
    }

    float out[nnet->outputSize];
    struct Matrix output = {out, nnet->outputSize, 1};

    forward_prop_conv(nnet, &adv, &output);

    int is_adv = 0;
    is_adv = check_functions_norm1(nnet, &output);

//    printMatrix(&adv);
//    printf("output is:\n");
//    printMatrix(&output);

    if (is_adv) {
        printf("\nadv found:\n");
        printf("adv is: ");
        printMatrix(&adv);
        denormalize_input(nnet, &adv);
        printf("adv is: ");
        printMatrix(&adv);
        printf("it's output is: ");
        printMatrix(&output);
        pthread_mutex_lock(&lock);
        adv_found = 1;
        pthread_mutex_unlock(&lock);
    }
    else // insert by ybh
    {
    	forward_prop_conv(nnet, &input->lower_matrix, &output);
    	 is_adv = check_functions_norm1(nnet, &output);
//    	    printf("output is:\n");
//    	    printMatrix(&output);
    	    if (is_adv) {
    	        printf("\nadv found:\n");
    	        printf("adv is: ");
    	        printMatrix(&adv);
    	        denormalize_input(nnet, &adv);
    	        printf("adv is: ");
    	        printMatrix(&adv);
    	        printf("it's output is: ");
    	        printMatrix(&output);
    	        pthread_mutex_lock(&lock);
    	        adv_found = 1;
    	        pthread_mutex_unlock(&lock);
    	    }
    	    else
    	    {
    	    	forward_prop_conv(nnet, &input->upper_matrix, &output);
    	    	 is_adv = check_functions_norm1(nnet, &output);
//    	    	    printf("output is:\n");
//    	    	    printMatrix(&output);
    	    	    if (is_adv) {
    	    	        printf("\nadv found:\n");
    	    	        printf("adv is: ");
    	    	        printMatrix(&adv);
    	    	        denormalize_input(nnet, &adv);
    	    	        printf("adv is: ");
    	    	        printMatrix(&adv);
    	    	        printf("it's output is: ");
    	    	        printMatrix(&output);
    	    	        pthread_mutex_lock(&lock);
    	    	        adv_found = 1;
    	    	        pthread_mutex_unlock(&lock);
    	    	    }
    	    	    else
    	    	    {
    	    	        for (int i=0;i<nnet->inputSize;i++) {
    	    	            float upper = input->upper_matrix.data[i];
    	    	            float lower = input->lower_matrix.data[i];
    	    	            float middle = lower/3+2*upper/3;

    	    	            a[i] = middle;
    	    	        }
    	    	        forward_prop_conv(nnet, &adv, &output);
    	    	        is_adv = check_functions_norm1(nnet, &output);
    	    	        if (is_adv) {
    	    	        	pthread_mutex_lock(&lock);
    	    	        	adv_found = 1;
    	    	        	pthread_mutex_unlock(&lock);
    	    	        }
    	    	     }
    	    }
    }

}


int choose_spliting_node2(struct NNet *network,
        float **value_upper, float **value_lower,
        int **states,int global_strategy, int *split_layer, int *split_node)
{
	struct NNet* nnet = network;
	int numLayers    = nnet->numLayers;
	int inputSize    = nnet->inputSize;
	int outputSize   = nnet->outputSize;
	int maxLayerSize = nnet->maxLayerSize;
	float max_temp=0,temp_pre, temp;
	int need_split=0;
	int flag=0;
	//global splitting strategy
	if(global_strategy==1)
	{
      for(int i=1;i<numLayers;i++)
    	for(int j=0;j<maxLayerSize;j++)
    	{
    		if(states[i][j]==1) //Note:states[1][0] match with node[1][0]
    		{

    			flag=1;
    			need_split=1;
				temp=value_upper[i][j]-value_lower[i][j];
				if(temp>=max_temp)
				{
					max_temp=temp;
					*split_layer=i;
					*split_node=j;
					fflush(stdout);
				}
    		}
    	}
	}
	else //local splitting strategy from lower layer to high layer
	{
	     for(int i=1;i<numLayers;i++)
	     {
	    	 for(int j=0;j<maxLayerSize;j++)
	    	{
	    		if(states[i][j]==1) //Note:states[0][0] match with node[1][0]
	    		{

	    			flag=1;
	    			need_split=1;
					temp=value_upper[i][j]-value_lower[i][j];
					if(temp>=max_temp)
					{
						max_temp=temp;
						*split_layer=i;
						*split_node=j;
						fflush(stdout);
					}
	    		}
	    	}
	    	 if(flag==1)
	    	 			break;
	     }
	}
	return need_split;
}


void *check_property_by_forward_backward_refining_thread(void *args)
{

    struct check_property_by_forward_backward_refining_args *actual_args = args;

    int ret=check_property_by_forward_backward_refining(actual_args->network,\
                    actual_args->input,\
                    actual_args->output,\
                    actual_args->value_upper,\
                    actual_args->value_lower,\
                    actual_args->symbol_upper,\
                    actual_args->symbol_lower,\
                    actual_args->states,\
		    actual_args->lp,\
		    actual_args->rule_num,\
		    actual_args->property,\
		    actual_args->need_prop,\
		    actual_args->depth,\
		    actual_args->split_layer,\
		    actual_args->change_split_layer,\
		    actual_args->input_refined,\
		    actual_args->output_map\
		    );
    return (void*)ret;

}


int split_by_predicates(
		struct NNet *network, struct Interval *input,
		 struct Interval *output,
		 float **value_upper, float **value_lower,
		 float **symbol_upper, float **symbol_lower, int **states,
		 lprec *lp, int *rule_num,int property,int split_layer,int split_node,int depth,int change_split_layer,int * output_map)
{
	depth+=1;
//   printf("sd:%d,\n",depth);
    pthread_mutex_lock(&lock);
    if(depth>m_depth)
	m_depth=depth;
    pthread_mutex_unlock(&lock);
	struct NNet* nnet=network;
	int inputSize=nnet->inputSize;
	int maxLayerSize=nnet->maxLayerSize;
	int numLayers=nnet->numLayers;
	int Ncol = inputSize;
	REAL row[Ncol+1];
	int ret;
	check_adv1(nnet,input);
	 pthread_mutex_lock(&lock);
	 if(adv_found==1)
         {pthread_mutex_unlock(&lock);
		 return 1;
	}
	pthread_mutex_unlock(&lock);

//	depth=depth+1;
    if (CHECK_ADV_MODE)
    {
        if (depth >= 24) {
        	printf("d: %d\n",depth);
            check_adv1(nnet, input);
            return 1;
        }
    }

    int depth1,depth2;
    depth1=depth;
    depth2=depth;

    int split_layer1,split_layer2,change_split_layer1,change_split_layer2;
    split_layer1=split_layer;
    split_layer2=split_layer;
    change_split_layer1=change_split_layer;
    change_split_layer1=change_split_layer;

    float input_upper1[nnet->inputSize];
    float input_lower1[nnet->inputSize];
    float input_upper2[nnet->inputSize];
    float input_lower2[nnet->inputSize];

    memcpy(input_upper1, input->upper_matrix.data,\
        sizeof(float)*inputSize);
    memcpy(input_upper2, input->upper_matrix.data,\
        sizeof(float)*inputSize);
    memcpy(input_lower1, input->lower_matrix.data,\
        sizeof(float)*inputSize);
    memcpy(input_lower2, input->lower_matrix.data,\
        sizeof(float)*inputSize);

    struct Interval input_interval = {
            (struct Matrix){input->lower_matrix.data, 1, nnet->inputSize},
            (struct Matrix){input->upper_matrix.data, 1, nnet->inputSize}
        };

    struct Interval input_interval1 = {
            (struct Matrix){input_lower1, 1, nnet->inputSize},
            (struct Matrix){input_upper1, 1, nnet->inputSize}
        };
    struct Interval input_interval2 = {
            (struct Matrix){input_lower2, 1, nnet->inputSize},
            (struct Matrix){input_upper2, 1, nnet->inputSize}
        };

    float o_upper1[nnet->outputSize], o_lower1[nnet->outputSize];
    memcpy(o_upper1, output->upper_matrix.data,\
           sizeof(float)*nnet->outputSize);
       memcpy(o_lower1, output->lower_matrix.data,\
           sizeof(float)*nnet->outputSize);
    struct Interval output_interval1 = {
            (struct Matrix){o_lower1, nnet->outputSize, 1},
            (struct Matrix){o_upper1, nnet->outputSize, 1}
        };

    float o_upper2[nnet->outputSize], o_lower2[nnet->outputSize];
    memcpy(o_upper2, output->upper_matrix.data,\
            sizeof(float)*nnet->outputSize);
        memcpy(o_lower2, output->lower_matrix.data,\
            sizeof(float)*nnet->outputSize);
    struct Interval output_interval2 = {
            (struct Matrix){o_lower2, nnet->outputSize, 1},
            (struct Matrix){o_upper2, nnet->outputSize, 1}
        };

    int **states1=(int **)malloc(sizeof(int*)*(nnet->numLayers+1));
    for(int i=0;i<nnet->numLayers+1;i++)
    {
    	states1[i]=(int*)malloc(sizeof(int)*nnet->maxLayerSize);
    	memcpy(states1[i], states[i], sizeof(int)*nnet->maxLayerSize);
    }
    int **states2=(int **)malloc(sizeof(int*)*(nnet->numLayers+1));
    for(int i=0;i<nnet->numLayers+1;i++)
    {
    	states2[i]=(int*)malloc(sizeof(int)*nnet->maxLayerSize);
    	memcpy(states2[i], states[i], sizeof(int)*nnet->maxLayerSize);
    }

    float **value_upper1=(float**)malloc(sizeof(float*)*(nnet->numLayers+1));
    float **value_lower1=(float**)malloc(sizeof(float*)*(nnet->numLayers+1));
	for(int i=0;i<nnet->numLayers+1;i++)
	{
		value_upper1[i]=(float*)malloc(sizeof(float)*nnet->maxLayerSize);
		memcpy(value_upper1[i], value_upper[i], sizeof(float)*nnet->maxLayerSize);
		value_lower1[i]=(float*)malloc(sizeof(float)*nnet->maxLayerSize);
		memcpy(value_lower1[i], value_lower[i], sizeof(float)*nnet->maxLayerSize);
//		struct Matrix temp= {(float*)value_upper[i],1,maxLayerSize};
//		printMatrix(&temp);
	}

    float **symbol_upper1=(float**)malloc(sizeof(float*)*(numLayers+1));
    float **symbol_lower1=(float**)malloc(sizeof(float*)*(numLayers+1));
    for(int i=0;i<numLayers+1;i++)
    {
    	symbol_upper1[i]=(float*)malloc(sizeof(float)*maxLayerSize*(inputSize+1));
    	memcpy(symbol_upper1[i], symbol_upper[i], sizeof(float)*maxLayerSize*(inputSize+1));
    	symbol_lower1[i]=(float*)malloc(sizeof(float)*maxLayerSize*(inputSize+1));
    	memcpy(symbol_lower1[i], symbol_lower[i], sizeof(float)*maxLayerSize*(inputSize+1));
    }
    int outputSize=nnet->outputSize;
    int output_map1[outputSize];
    int output_map2[outputSize];
    memcpy(output_map1, output_map, sizeof(int)*outputSize);
    memcpy(output_map2, output_map, sizeof(int)*outputSize);

	lprec *lp1, *lp2;
	lp1 = copy_lp(lp);
	lp2 = copy_lp(lp);

	int rule_num1 = *rule_num;
	int rule_num2 = *rule_num;

	memset(row, 0, Ncol*sizeof(float));

	//splitting with cons<=0
	set_add_rowmode(lp1, TRUE);
//	int flag_up_eqs_low=0;
	for(int j=1;j<Ncol+1;j++){
		row[j] = symbol_upper[split_layer][j+split_node*(inputSize+1)-1];
	}

    add_constraint(lp1, row, LE,-symbol_upper[split_layer][inputSize+split_node*(inputSize+1)]);

 	rule_num1 += 1;

	set_add_rowmode(lp1, FALSE);
	set_presolve(lp1,PRESOLVE_ROWS+PRESOLVE_COLS,0);
	states1[split_layer][split_node]=0;

//splitting with cons>=0
	 set_add_rowmode(lp2, TRUE);
	 add_constraint(lp2, row, GE,-symbol_upper[split_layer][inputSize+split_node*(inputSize+1)]);
	 rule_num2 += 1;
	 set_add_rowmode(lp2, FALSE);
	 set_presolve(lp2,PRESOLVE_ROWS+PRESOLVE_COLS,0);
	 states2[split_layer][split_node]=2;


    for(int j=1;j<Ncol+1;j++){
        row[j] = 0;
    }


	int unsat1=0, unsat2=0;;
    int result1=0,result2=0;
	int need_check_adv1=0,need_check_adv=0;
    int flag=0;

	{
	    if(depth%15==0)
		{

		//计算第i维的上下界。
		   set_timeout(lp2,10);
		   for(int i=0;i<inputSize;i++)
		   {
			for(int j=1;j<Ncol+1;j++){
		      		row[j] = 0;
		   	}
			int refined=0;
			row[i+1]=1;
			set_obj_fn(lp2, row);
			set_maxim(lp2);
			ret = solve(lp2);
			if(ret == 0)
			{
		             float temp= get_objective(lp2);
			     if(temp<input_interval2.upper_matrix.data[i])
	    		     {
			  	refined=1;
			  	input_interval2.upper_matrix.data[i]=temp;
			     }
			}
		        else{
		           result2 = 1;
		           break;
		        }
			set_minim(lp2);
			ret = solve(lp2);
			if(ret == 0){
		            float temp= get_objective(lp2);
			    if(temp>input_interval2.lower_matrix.data[i])
	    		    {
			 	 refined=1;
	    			 input_interval2.lower_matrix.data[i]=temp;
			    }
			}
		        else{
		       	    result2 = 1;
		            break;
		        }
		        row[i+1]=0;
			if(refined==1&&unsat2==0)
			{
				need_check_adv=1;
		//	printf("it is refined1!\n");
			     if(input_interval2.lower_matrix.data[i]>input_interval2.upper_matrix.data[i])
				input_interval2.upper_matrix.data[i]=input_interval2.lower_matrix.data[i];
		             set_bounds(lp2,i+1,input_interval2.lower_matrix.data[i],input_interval2.upper_matrix.data[i]);
			}
		    }//end for
		}
	}

	{

	    	if(depth%15==0)
	    	{
		    set_timeout(lp1,10);
		    for(int i=0;i<inputSize;i++)
		    {
			for(int j=1;j<Ncol+1;j++){
			      row[j] = 0;
			   }
			int refined=0;
			row[i+1]=1;
			set_obj_fn(lp1, row);
			set_maxim(lp1);
			ret = solve(lp1);
			if(ret == 0){
		            float temp= get_objective(lp1);
			    if(temp<input_interval1.upper_matrix.data[i])
	    	            {
		        	refined=1;
		  		input_interval1.upper_matrix.data[i]=temp;
			    }
			}
		        else{
		            result1 = 1;
		            break;
		    	}
			set_minim(lp1);
			ret = solve(lp1);
			if(ret == 0){
		            float temp= get_objective(lp1);
		       	    if(temp>input_interval1.lower_matrix.data[i])
	    		    {
			  	refined=1;
			  	input_interval1.lower_matrix.data[i]=temp;
			    }
			}
		        else{
		            result1 = 1;
		            break;
		        }
			if(unsat1==0 && refined==1)
		        {
				need_check_adv1=1;
			//	printf("it is refined!\n");
				if(input_interval1.lower_matrix.data[i]>input_interval1.upper_matrix.data[i])
					input_interval1.upper_matrix.data[i]=input_interval1.lower_matrix.data[i];
		       	        set_bounds(lp1,i+1,input_interval1.lower_matrix.data[i],input_interval1.upper_matrix.data[i]);
			}
			row[i+1]=0;
		    }
		}

	}

	 pthread_mutex_lock(&lock);
//        if(0){
	 if(count<=MAX_THREAD) {
           pthread_mutex_unlock(&lock);
           pthread_t workers1, workers2;
       struct check_property_by_forward_backward_refining_args args1 = {nnet,&input_interval1,&output_interval1,value_upper,value_lower,symbol_upper,symbol_lower,states1,\
           	         		lp1,&rule_num1,property,1,depth1,split_layer1,change_split_layer1,need_check_adv1,output_map1};

        struct check_property_by_forward_backward_refining_args args2 = {nnet,&input_interval2,&output_interval2,value_upper1,value_lower1,symbol_upper1,symbol_lower1,states2,\
             		lp2,&rule_num2,property,1,depth2,split_layer2,change_split_layer2,need_check_adv,output_map2};
	if(result1==0)
        {
		pthread_create(&workers1, NULL, check_property_by_forward_backward_refining_thread, &args1);
	        pthread_mutex_lock(&lock);
	        count++;

	        pthread_mutex_unlock(&lock);
	}

	if(result2==0)
        {
		pthread_create(&workers2, NULL, check_property_by_forward_backward_refining_thread, &args2);
        	pthread_mutex_lock(&lock);
        	count++;
   		pthread_mutex_unlock(&lock);
	}
        void* result21=result1;
        void* result22=result2;

    	if(result2==0)
    	{
            	pthread_join(workers2, &result22);
            	pthread_mutex_lock(&lock);
            	count--;
    		pthread_mutex_unlock(&lock);
    	}
    	  result2=(int)result22;

    	    delete_lp(lp2);
    	        	for(int i=0;i<nnet->numLayers+1;i++)
    	        	{
    	        		free(value_upper1[i]);
    	        		free(value_lower1[i]);
    	            	free(symbol_upper1[i]);
    	            	free(symbol_lower1[i]);
    	            	free(states2[i]);
    	        	}
    	        	free(value_upper1);
    	        	free(value_lower1);
    	            free(symbol_upper1);
    	            free(symbol_lower1);
    	              free(states2);

	if(result1==0)
        {
		pthread_join(workers1, &result21);
	        pthread_mutex_lock(&lock);
        	count--;
        	pthread_mutex_unlock(&lock);
       }

    delete_lp(lp1);
	for(int i=0;i<nnet->numLayers+1;i++)
    {
        free(states1[i]);
    }
    free(states1);


	result1=(int)result21;

       }
       else
       {
         pthread_mutex_unlock(&lock);


   	  if(result2==0)
             result2 = check_property_by_forward_backward_refining(nnet,&input_interval2,&output_interval2,value_upper1,value_lower1,symbol_upper1,symbol_lower1,states2,\
                		lp2,&rule_num2,property,1,depth2,split_layer2,change_split_layer2,need_check_adv,output_map2);


      delete_lp(lp2);
          	for(int i=0;i<nnet->numLayers+1;i++)
          	{
          		free(value_upper1[i]);
          		free(value_lower1[i]);
              	free(symbol_upper1[i]);
              	free(symbol_lower1[i]);
              	free(states2[i]);
          	}
          	free(value_upper1);
          	free(value_lower1);
              free(symbol_upper1);
              free(symbol_lower1);
                free(states2);


	 if(result1==0)
           result1=check_property_by_forward_backward_refining(nnet,&input_interval1,&output_interval1,value_upper,value_lower,symbol_upper,symbol_lower,states1,\
           	         		lp1,&rule_num1,property,1,depth1,split_layer1,change_split_layer1,need_check_adv1,output_map1);

	    delete_lp(lp1);
		for(int i=0;i<nnet->numLayers+1;i++)
	    {
	        free(states1[i]);
	    }
	    free(states1);
       }
    if(result1==1&&result2==1)
    	return 1;
    else
    	return 0;
}
