#include "split.h"

#define AVG_WINDOW 5
//#define MAX_THREAD 128
#define MIN_DEPTH_PER_THREAD 5 

int NEED_PRINT = 0;
int NEED_FOR_ONE_RUN = 0;

int input_depth = 0;
int adv_found = 0;
int forward_count=0;
int m_depth =0;
int count = 0;
int thread_tot_cnt  = 0;

int RELAX_MOD=0;//0 for ours, 1 for POPL, 2 for Neurify
int FUL_or_RAND=0;// 0 for FUL , 1 for random
int MAX_THREAD= 128;//
int REFINE_delay_NUM=1; // 0 for no refinement, default:1
int SPLITING_METHOD=2; // 0 for dependency, 1 for smallest value range, 2 for largest value range, 3 for smallest index first, 4 for non_ful strategy

int progress = 0;

int CHECK_ADV_MODE = 0;
int PARTIAL_MODE = 0;

int DEBUG = 0;

float avg_depth = 50;
float total_avg_depth = 0;
int leaf_num = 0;
float max_depth = 0;

struct timeval start,finish,last_finish,start1,finish1;

// 1 for verified, 0 for not verified.
int check_max_constant(struct NNet *nnet, struct Interval *output)
{
//	printf("the output of target is%f\n",output->upper_matrix.data[nnet->target]);
	if (output->upper_matrix.data[nnet->target] > 0.5011) {
        return 0;
    }
    else {
        return 1;
    }
}

//coc is not the maximal score
// 1 for verified, 0 for not verified.
int check_max(struct NNet *nnet, struct Interval *output)
{
//    printf("run check_max\n");
    for (int i=0;i<nnet->outputSize;i++) {
        if (output->lower_matrix.data[i]>0 && i != nnet->target) {
            return 1;
        }

    }
    return 0;
}
// 1 for verified, 0 for not verified.
int check_min(struct NNet *nnet, struct Interval *output)
{
    for (int i=0;i<nnet->outputSize;i++) {

        if (output->upper_matrix.data[i]<0 && i != nnet->target) {
            return 1;
        }

    }

    return 0;
}

int check_not_min(struct NNet *nnet, struct Interval *output)
{

    for (int i=0;i<nnet->outputSize;i++) {

        if (output->lower_matrix.data[i]<0 && i != nnet->target) {
            return 0;
        }

    }

    return 1;
}

int check_max_constant1(struct NNet *nnet, struct Matrix *output)
{

    if (output->data[nnet->target] < 0.50012) {
        return 0;
    }

    return 1;
}



int check_max1(struct NNet *nnet, struct Matrix *output)
{

    for (int i=0;i<nnet->outputSize;i++) {

        if (output->data[i] > 0 && i != nnet->target) {
            return 0;
        }

    }

    return 1;

}


int check_min1(struct NNet *nnet, struct Matrix *output)
{

    for (int i=0;i<nnet->outputSize;i++) {

        if (output->data[i] < 0 && i != nnet->target) {
            return 0;
        }

    }

    return 1;

}


int check_not_min1(struct NNet *nnet, struct Matrix *output)
{

    for (int i=0;i<nnet->outputSize;i++) {

        if (output->data[i] < 0 && i != nnet->target) {
            return 1;
        }

    }

    return 0;
}

int check_max_constant_with_constraints(struct NNet *nnet, struct Interval *output,float **symbol_upper, float **symbol_lower,lprec *lp) // 1 for verified, 0 for not verified.
{
//	printf("the output of target is%f\n",output->upper_matrix.data[nnet->target]);
	if (output->upper_matrix.data[nnet->target] > 0.5011) {
		int inputSize=nnet->inputSize;
		int numLayers  = nnet->numLayers;
		int target = nnet->target;
		int Ncol = inputSize;
		int ret;
		float precise_output_upper=10;
// /*
        REAL row[Ncol+1];
		//计算第i维的上下界。
		set_timeout(lp, 10);
		for (int j = 1; j < Ncol + 1; j++) {
			row[j] = symbol_upper[numLayers][j + target * (inputSize + 1) - 1];
		}
		set_obj_fn(lp, row);
		set_maxim(lp);
//		   	write_LP(lp,stdout);
		ret = solve(lp);
//    	  printf("the ret is %d!\n", ret);
		if (ret == 0) {
			precise_output_upper = get_objective(lp)+symbol_upper[numLayers][inputSize+target*(inputSize+1)];
		}
//		*/
//	    if(precise_output_upper>output->upper_matrix.data[nnet->target]+0.000001)
//	     	printf("error,the output of target is%f, and more precise %f\n",output->upper_matrix.data[nnet->target],precise_output_upper);
		if(precise_output_upper > 0.5011)
			return 0;
		else
			return 1;
    }
    else {
        return 1;
    }

}

//coc is not the maximal score
int check_max_with_constraints(struct NNet *nnet, struct Interval *output,float **symbol_upper, float **symbol_lower,lprec *lp)
{
//    printf("run check_max\n");
    for (int i=0;i<nnet->outputSize;i++) {

    	if(i != nnet->target)
    	{
        if (output->lower_matrix.data[i]>0) {
            return 1;
        }
        else
        {
    		int inputSize=nnet->inputSize;
    		int numLayers  = nnet->numLayers;
    		int target = nnet->target;
    		int Ncol = inputSize;
    		int ret;
    		float precise_output_lower=-10;
    // /*
            REAL row[Ncol+1];
    		//计算第i维的上下界。
    		set_timeout(lp, 10);
    		for (int j = 1; j < Ncol + 1; j++) {
    			row[j] = symbol_lower[numLayers][j + i * (inputSize + 1) - 1];
    		}
    		set_obj_fn(lp, row);
    		set_minim(lp);
    //		   	write_LP(lp,stdout);
    		ret = solve(lp);
    //    	  printf("the ret is %d!\n", ret);
    		if (ret == 0) {
    			precise_output_lower = get_objective(lp)+symbol_lower[numLayers][inputSize+i*(inputSize+1)];
    		}
        	if(precise_output_lower>0)
        		return 1;
        }
    	}

    }

    return 0;

}


int check_min_with_constraints(struct NNet *nnet, struct Interval *output,float **symbol_upper, float **symbol_lower,lprec *lp)
{
    for (int i=0;i<nnet->outputSize;i++) {
    	if(i != nnet->target)
    	{
        if (output->upper_matrix.data[i]<0) {
            return 1;
        }
        else
        {
    		int inputSize=nnet->inputSize;
    		int numLayers  = nnet->numLayers;
    		int target = nnet->target;
    		int Ncol = inputSize;
    		int ret;
    		float precise_output_upper=10;
    // /*
            REAL row[Ncol+1];
    		//计算第i维的上下界。
    		set_timeout(lp, 10);
    		for (int j = 1; j < Ncol + 1; j++) {
    			row[j] = symbol_upper[numLayers][j + i * (inputSize + 1) - 1];
    		}
    		set_obj_fn(lp, row);
    		set_maxim(lp);
    //		   	write_LP(lp,stdout);
    		ret = solve(lp);
    //    	  printf("the ret is %d!\n", ret);
  		if (ret == 0) {
    			precise_output_upper = get_objective(lp)+symbol_upper[numLayers][inputSize+i*(inputSize+1)];
    		}
 //        	printf("error,the output of target is%f, and more precise %f\n",output->upper_matrix.data[i],precise_output_upper);

        	if(precise_output_upper<0)
        		return 1;
        }
    	}
    }
    return 0;
}


int check_functions(struct NNet *nnet, struct Interval *output)
{
    if (PROPERTY ==1) {
        return check_max_constant(nnet, output);
    }

    if (PROPERTY == 2) {
        return check_max(nnet, output);
    }

    if (PROPERTY == 3) {
    	return check_min(nnet, output);
    }

    if (PROPERTY == 4) {
        return check_min(nnet, output);
    }
    return -1;

}

int check_functions_wit_constraints(struct NNet *nnet, struct Interval *output,float **symbol_upper, float **symbol_lower,lprec *lp)
{

    if (PROPERTY ==1) {
    	return check_max_constant(nnet, output);
    }
    if (PROPERTY == 2) {
        return check_max_with_constraints(nnet, output,symbol_upper,symbol_lower,lp);
    }

    if (PROPERTY == 3) {
//        return check_min(nnet, output);
    	return check_min_with_constraints(nnet, output,symbol_upper,symbol_lower,lp);
    }

    if (PROPERTY == 4) {
        return check_min(nnet, output);
    }
    return -1;

}


int check_functions1(struct NNet *nnet, struct Matrix *output)
{

    if (PROPERTY == 1) {
        return check_max_constant1(nnet, output);
    }

    if (PROPERTY == 2) {
        return check_max1(nnet, output);
    }

    if (PROPERTY == 3) {
        return check_min1(nnet, output);
    }

    if (PROPERTY == 4) {
        return check_min1(nnet, output);
    }
    return -1;


}

void *check_property_by_forward_backward_refining_thread(void *args)
{

    struct check_property_by_forward_backward_refining_args *actual_args = args;

    int ret=check_property_by_forward_backward_refining(actual_args->network,\
                    actual_args->input,\
                    actual_args->value_upper,\
                    actual_args->value_lower,\
                    actual_args->symbol_upper,\
                    actual_args->symbol_lower,\
                    actual_args->states,\
        		    actual_args->equation_upper,\
        		    actual_args->equation_lower,\
                            actual_args->new_equation_upper,\
        		    actual_args->new_equation_lower,\
		    actual_args->lp,\
		    actual_args->rule_num,\
		    actual_args->property,\
		    actual_args->need_prop,\
		    actual_args->depth,\
		    actual_args->split_layer,\
		    actual_args->change_split_layer,\
		    actual_args->input_refined
);

    return (void*)ret;

}

void check_adv(struct NNet* nnet, struct Interval *input)
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

    forward_prop(nnet, &adv, &output);

    int is_adv = 0;
    is_adv = check_functions1(nnet, &output);

    //printMatrix(&adv);
    //printMatrix(&output);

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
    	 forward_prop(nnet, &input->lower_matrix, &output);
    	 is_adv = check_functions1(nnet, &output);
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
    	    	 forward_prop(nnet, &input->upper_matrix, &output);
    	    	 is_adv = check_functions1(nnet, &output);
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
    	    }
    }

}

void compute_extremums(int inputSize,struct Interval *input,float *equation_dep0, float bias_dep0,float *equation_dep1, float bias_dep1, float *min, float *max)
{
    int nonzero_index = 0;
    while (equation_dep0[nonzero_index] == 0)
        nonzero_index += 1;
    float new_equation[inputSize];
    float new_offset=0;

    float tempVal_lower=0,tempVal_upper=0;
    for(int i=0;i<inputSize;i++)
    {
    	new_equation[i] = equation_dep1[i] - equation_dep1[nonzero_index]/equation_dep0[nonzero_index]*equation_dep0[i];
    }

    new_offset =bias_dep1 - equation_dep1[nonzero_index]/equation_dep0[nonzero_index]*bias_dep0;

    for(int k=0;k<inputSize;k++)
    {
          if(new_equation[k]>=0){
              tempVal_lower = tempVal_lower + new_equation[k] * input->lower_matrix.data[k];
              tempVal_upper = tempVal_upper + new_equation[k] * input->upper_matrix.data[k];
          }
          else{
        	  tempVal_lower = tempVal_lower + new_equation[k] * input->upper_matrix.data[k];
        	  tempVal_upper = tempVal_upper + new_equation[k] * input->lower_matrix.data[k];
          }

     }
    *min = tempVal_lower + new_offset;
    *max = tempVal_upper + new_offset;
}

int dependency_between_two_nodes(int inputSize,struct Interval *input,float *equation_dep0, float bias_dep0,float *equation_dep1, float bias_dep1)
{
	int dep=0;
	float min0=0, max0=0, min1=0, max1=0;
	compute_extremums(inputSize,input,equation_dep0,bias_dep0,equation_dep1,bias_dep1,&min0,&max0);
	compute_extremums(inputSize,input,equation_dep1,bias_dep1,equation_dep0,bias_dep0,&min1,&max1);
  //  printf("min0 max0 min1 max1 are %f,%f, %f,%f\n",min0,max0,min1,max1);
    if(max0 == 0.0 && max1 == 0.0)
    {
    	printf("oh,good!\n");
    	dep=2;
    }
	if(max0 < 0 && max1 < 0)
		dep=1;
	if(min0 > 0 && min1 > 0)
		dep=1;
	if(max0 < 0 && min1 > 0)
		dep=1;
	if(min0 > 0 && max1 < 0)
		dep=1;
	return dep;
}

int choose_spliting_node_by_dependency(struct NNet *network,struct Interval *input,
     //   float **value_upper_pre, float **value_lower_pre,
        float **value_upper, float **value_lower,
        float **symbol_upper, float **symbol_lower,
        int **states, int *split_layer, int *split_node)
{

	printf("input ranges:\n");
			printMatrix(&input->upper_matrix);
		        printMatrix(&input->lower_matrix);

	struct NNet* nnet = network;
	int numLayers    = nnet->numLayers;
	int inputSize    = nnet->inputSize;
	int outputSize   = nnet->outputSize;
	int maxLayerSize = nnet->maxLayerSize;
	float max_temp=0,temp_pre, temp;
	int need_split=0;
	int res_two_node=0;
    int res[maxLayerSize];
    for(int i=0;i<maxLayerSize;i++)
    	res[i]=0;

 //   int num_non_sta=0;
    float equation_dep0[inputSize], equation_dep1[inputSize];
    float bias_dep0,bias_dep1;

//    float temp=0,max_temp=0;

	int flag=0;
     for(int i=1;i<numLayers;i++)
     {
    	 for(int j=0;j<maxLayerSize;j++)
    	{

    		if(states[i][j]==1) //Note:states[0][0] match with node[1][0]
    		{
    			flag=1;
    			need_split=1;
				*split_layer=i;
				*split_node=j;
//	    			printf("The value_upper_pre and value_lower_pre are %f,%f\n",value_upper_pre[i][j],value_lower_pre[i][j]);
//	    			printf("The value_upper and value_lower are %f,%f\n",value_upper[i][j],value_lower[i][j]);
    			 for(int m=0;m<inputSize;m++){
    				 equation_dep0[m] = symbol_lower[i][m+j*(inputSize+1)];
//    				 printf("inputSize is %d,The symbol_upper and symbol_lower of node[%d][%d] are %f,%f\n",inputSize,i,j,symbol_lower[i][m+j*(inputSize+1)],symbol_upper[i][m+j*(inputSize+1)]);

    			 }
    			 bias_dep0=symbol_lower[i][inputSize+j*(inputSize+1)];
 //   			 printf("bias of node[%d][%d] are %f\n",i,j,bias_dep0);


    			for(int k=j+1;k<maxLayerSize;k++)
                {
                	if(states[i][k]==1)
                	{

                		 for(int m=0;m<inputSize;m++){
                	    	equation_dep1[m] = symbol_lower[i][m+k*(inputSize+1)];
//                	   	 printf("inputSize is %d,The symbol_upper and symbol_lower of node[%d][%d] are %f,%f\n",inputSize,i,k,symbol_lower[i][m+k*(inputSize+1)],symbol_upper[i][m+k*(inputSize+1)]);

                	     }
                	     bias_dep1=symbol_lower[i][inputSize+k*(inputSize+1)];
//                	     printf("bias of node[%d][%d] are %f\n",i,k,bias_dep1);

                		res_two_node=dependency_between_two_nodes(inputSize,input,equation_dep0,bias_dep0,equation_dep1,bias_dep1);

                	}
                	if(res_two_node==1)
                	{
                		res[j]+=1;
                		res[k]+=1;
                	}
                }
    		}
    	}
		int max_dep=res[0];
		for(int t=0;t<maxLayerSize;t++)
		{
			printf("The dep num of node[%d][%d] is %d\n",i,t,res[t]);


    //        if(i>=2)
            {
			if(res[t]>max_dep && states[i][t]==1)
			{
				max_dep=res[t];
				*split_layer=i;
				*split_node=t;
			}
			else if(res[t]==max_dep && states[i][t]==1)
			{
				//consider value range larger first?

				temp=value_upper[i][t]-value_lower[i][t];
	//				printf("the temp_pre is %f",temp_pre);

				//float rate=(temp_pre-temp)/temp_pre;
	//					printf("the layer and node are here now, %d,%d, the rate is%f.\n",i,j,rate);
				if(temp>=max_temp)
				{
					max_temp=temp;
					*split_layer=i;
					*split_node=t;
	//				printf("the layer and node are here, %d,%d\n",i,j);
					fflush(stdout);
				}
			}
            }
/*            else if(states[i][t]==1)
            {
				temp=value_upper[i][t]-value_lower[i][t];
	//				printf("the temp_pre is %f",temp_pre);

				//float rate=(temp_pre-temp)/temp_pre;
	//					printf("the layer and node are here now, %d,%d, the rate is%f.\n",i,j,rate);
				if(temp>=max_temp)
				{
					max_temp=temp;
					*split_layer=i;
					*split_node=t;
	//				printf("the layer and node are here, %d,%d\n",i,j);
					fflush(stdout);
				}
            }
            */
		}
	    if(flag==1)
			break;
     }
     return need_split;
}


/*
//the value_range changing rate largest
int choose_spliting_node(struct NNet *network,
        float **value_upper_pre, float **value_lower_pre,
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
	//global splitting strategy
	if(global_strategy==1)
	{
      for(int i=1;i<numLayers;i++)
    	for(int j=0;j<maxLayerSize;j++)
    	{
//    		printf("states[%d][%d] is %d\n",i-1,j,states[i-1][j]);
    		if(states[i][j]==1) //Note:states[1][0] match with node[1][0]
    		{
    			need_split=1;
//    			printf("The value_upper_pre and value_lower_pre are %f,%f\n",value_upper_pre[i][j],value_lower_pre[i][j]);
				temp_pre=value_upper_pre[i][j]-value_lower_pre[i][j];
				temp=value_upper[i][j]-value_lower[i][j];
//				printf("the temp_pre is %f",temp_pre);
				fflush(stdout);
				if(temp_pre==0)
				{   printf("div_by_zero.\n");
					*split_layer=i;
					*split_node=j;
					break;
				}
				float rate=(temp_pre-temp)/temp_pre;
				if(rate>=max_temp)
				{
					max_temp=rate;
					*split_layer=i;
					*split_node=j;
				//	printf("the layer and node are here, %d,%d",i,j);
				}
    		}
    	}
	}
	else //local splitting strategy from lower layer to high layer
	{
		int flag=0;
	     for(int i=1;i<numLayers;i++)
	     {
	    	 for(int j=0;j<maxLayerSize;j++)
	    	{
//	    		printf("states[%d][%d] is %d\n",i-1,j,states[i-1][j]);
	    		if(states[i][j]==1) //Note:states[0][0] match with node[1][0]
	    		{

	    			flag=1;
	    			need_split=1;
//	    			printf("The value_upper_pre and value_lower_pre are %f,%f\n",value_upper_pre[i][j],value_lower_pre[i][j]);
//	    			printf("The value_upper and value_lower are %f,%f\n",value_upper[i][j],value_lower[i][j]);

	    	//		temp_pre=value_upper_pre[i][j]-value_lower_pre[i][j];
					temp=value_upper[i][j]-value_lower[i][j];
	//				printf("the temp_pre is %f",temp_pre);
					fflush(stdout);
					if(temp_pre==0)
					{   printf("div_by_zero.\n");
						*split_layer=i;
						*split_node=j;
						break;
					}
					float rate=(temp_pre-temp)/temp_pre;
//					printf("the layer and node are here now, %d,%d, the rate is%f.\n",i,j,rate);
					if(rate>=max_temp)
					{
						max_temp=rate;
						*split_layer=i;
						*split_node=j;
		//				printf("the layer and node are here, %d,%d\n",i,j);
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
*/

//largest_value_range_first
int choose_spliting_node2(struct NNet *nnet,
//        float **value_upper_pre, float **value_lower_pre,
        float **value_upper, float **value_lower,
        int **states,int *split_layer, int *split_node)
{
//	struct NNet* nnet = network;
	int numLayers    = nnet->numLayers;
	int inputSize    = nnet->inputSize;
	int outputSize   = nnet->outputSize;
	int maxLayerSize = nnet->maxLayerSize;
	float max_temp=0,temp_pre, temp;
	int need_split=0;
	int flag=0;
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
					}
	    		}
	    	}
	    	 if(flag==1)
	    	 			break;
	     }
	return need_split;
}

//smallest_value_range_first
int choose_spliting_node3(struct NNet *network,
 //       float **value_upper_pre, float **value_lower_pre,
        float **value_upper, float **value_lower,
        int **states,int global_strategy, int *split_layer, int *split_node)
{
	struct NNet* nnet = network;
	int numLayers    = nnet->numLayers;
	int inputSize    = nnet->inputSize;
	int outputSize   = nnet->outputSize;
	int maxLayerSize = nnet->maxLayerSize;
	float min_temp=0,temp_pre, temp,max_temp=0;
	int need_split=0;
	int flag=0;
	//global splitting strategy
	if(global_strategy==1)
	{
      for(int i=1;i<numLayers;i++)
    	for(int j=0;j<maxLayerSize;j++)
    	{
//    		printf("states[%d][%d] is %d\n",i-1,j,states[i-1][j]);
    		if(states[i][j]==1) //Note:states[1][0] match with node[1][0]
    		{
    			need_split=1;
//	    			printf("The value_upper_pre and value_lower_pre are %f,%f\n",value_upper_pre[i][j],value_lower_pre[i][j]);
//	    			printf("The value_upper and value_lower are %f,%f\n",value_upper[i][j],value_lower[i][j]);

//	    			temp_pre=value_upper_pre[i][j]-value_lower_pre[i][j];
				temp=value_upper[i][j]-value_lower[i][j];
                                    if(flag==0)
                                    	 min_temp=temp;
//				printf("the temp_pre is %f",temp_pre);
				if(temp<=min_temp)
				{
					min_temp=temp;
					*split_layer=i;
					*split_node=j;
	//				printf("the layer and node are here, %d,%d\n",i,j);
					fflush(stdout);
				}
                           flag=1;
    		}
    	}
	}
	else //local splitting strategy from lower layer to high layer
	{
	     for(int i=1;i<numLayers;i++)
	     {
	    	 for(int j=0;j<maxLayerSize;j++)
	    	{
//	    		printf("states[%d][%d] is %d\n",i-1,j,states[i-1][j]);
	    		if(states[i][j]==1) //Note:states[0][0] match with node[1][0]
	    		{


	    			
	    			need_split=1;
//	    			printf("The value_upper_pre and value_lower_pre are %f,%f\n",value_upper_pre[i][j],value_lower_pre[i][j]);
//	    			printf("The value_upper and value_lower are %f,%f\n",value_upper[i][j],value_lower[i][j]);

//	    			temp_pre=value_upper_pre[i][j]-value_lower_pre[i][j];
					temp=value_upper[i][j]-value_lower[i][j];
                                        if(flag==0)
                                        	 min_temp=temp;
	//				printf("the temp_pre is %f",temp_pre);
					if(temp<=min_temp)
					{
						min_temp=temp;
						*split_layer=i;
						*split_node=j;
		//				printf("the layer and node are here, %d,%d\n",i,j);
						fflush(stdout);
					}
                               flag=1;
	    		}
	    	}
	    	 if(flag==1)
	    	 			break;
	     }
	}
	return need_split;
}

//first_non_det_first
int choose_spliting_node4(struct NNet *network,
//        float **value_upper_pre, float **value_lower_pre,
        float **value_upper, float **value_lower,
        int **states,int global_strategy, int *split_layer, int *split_node)
{
	struct NNet* nnet = network;
	int numLayers    = nnet->numLayers;
	int inputSize    = nnet->inputSize;
	int outputSize   = nnet->outputSize;
	int maxLayerSize = nnet->maxLayerSize;
	float min_temp=0,temp_pre, temp,max_temp=0;
	int need_split=0;
	int flag=0;
	//global splitting strategy
	if(global_strategy==1)
	{
      for(int i=1;i<numLayers;i++)
      {
	    	 for(int j=0;j<maxLayerSize;j++)
	    	{
//	    		printf("states[%d][%d] is %d\n",i-1,j,states[i-1][j]);
	    		if(states[i][j]==1) //Note:states[0][0] match with node[1][0]
	    		{

	    			need_split=1;
//
						*split_layer=i;
						*split_node=j;
		//				printf("the layer and node are here, %d,%d\n",i,j);
						fflush(stdout);
                            flag=1;
			       break;
	    		}
	    	}
	    	 if(flag==1)
	    	 			break;
      }
	}
	else //local splitting strategy from lower layer to high layer
	{

	     for(int i=1;i<numLayers;i++)
	     {
	    	 for(int j=0;j<maxLayerSize;j++)
	    	{
//	    		printf("states[%d][%d] is %d\n",i-1,j,states[i-1][j]);
	    		if(states[i][j]==1) //Note:states[0][0] match with node[1][0]
	    		{

	    			need_split=1;
//	    	
						*split_layer=i;
						*split_node=j;
		//				printf("the layer and node are here, %d,%d\n",i,j);
						fflush(stdout);
                               flag=1;
			       break;
	    		}
	    	}
	    	 if(flag==1)
	    	 			break;
	     }
	}
	return need_split;
}

int choose_spliting_node5(struct NNet *network,
        float **value_upper, float **value_lower,
        int **states,int global_strategy, int *split_layer, int *split_node)
{
	printf("run choose_spliting_node\n");
	struct NNet* nnet = network;
	int numLayers    = nnet->numLayers;
	int inputSize    = nnet->inputSize;
	int outputSize   = nnet->outputSize;
	int maxLayerSize = nnet->maxLayerSize;
	float max_temp=0,temp_pre, temp;
	int need_split=0;

        int total_unsplit_node=0;
	int temp_count=0;
        for(int i=1;i<numLayers;i++)
	     {
	    	 for(int j=0;j<maxLayerSize;j++)
	    	{
	    		//printf("states1[%d][%d] is %d\n",i,j,states[i][j]);
	    		if(states[i][j]==1) //Note:states[0][0] match with node[1][0]
	    		{
			    total_unsplit_node++;
			}
		}
	}

	int split_num;
	if(total_unsplit_node==0)
            return need_split;
	split_num=rand()%total_unsplit_node;
	{
		int flag=0;
	     for(int i=1;i<numLayers;i++)
	     {
	    	 for(int j=0;j<maxLayerSize;j++)
	    	{
	    		//printf("states1[%d][%d] is %d\n",i,j,states[i][j]);
	    		if(states[i][j]==1) //Note:states[0][0] match with node[1][0]
	    		{
				if(temp_count==split_num)
				{
					flag=1;
	    			        need_split=1;
					*split_layer=i;
					*split_node=j;
				    break;
				}
				temp_count++;
	    		}
	    	}
	    	 if(flag==1)
	    	 			break;
	     }
	}
	if(need_split==1)
	   states[*split_layer][*split_node]=3;
	//printf("states[%d][%d] is %d\n",*split_layer,*split_node,states[*split_layer][*split_node]);
	return need_split;
}



int split_by_predicates(
		struct NNet *nnet, struct Interval *input,
//		 struct Interval *output,
		 float **value_upper, float **value_lower,
		 float **symbol_upper, float **symbol_lower, int **states,
		 float *equation_upper, float *equation_lower,
		 float *new_equation_upper, float *new_equation_lower,
		 lprec *lp, int *rule_num,int property,int split_layer,int split_node,int depth,int change_split_layer)
{
	depth+=1;
    pthread_mutex_lock(&lock);
    if(depth>m_depth)
	m_depth=depth;
    pthread_mutex_unlock(&lock);
	int inputSize=nnet->inputSize;
	int maxLayerSize=nnet->maxLayerSize;
	int numLayers=nnet->numLayers;
	int Ncol = inputSize;
	REAL row[Ncol+1];
	int ret;

	if(property!=1)
	{
	check_adv(nnet,input);
	 pthread_mutex_lock(&lock);
	 if(adv_found==1)
         {pthread_mutex_unlock(&lock);
		 return 1;
	}
	pthread_mutex_unlock(&lock);
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

    struct Interval input_interval1 = {
            (struct Matrix){input_lower1, 1, nnet->inputSize},
            (struct Matrix){input_upper1, 1, nnet->inputSize}
        };
    struct Interval input_interval2 = {
            (struct Matrix){input_lower2, 1, nnet->inputSize},
            (struct Matrix){input_upper2, 1, nnet->inputSize}
        };
    /*
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

*/


    int **states1=(int **)malloc(sizeof(int*)*(nnet->numLayers+1));
    int **states2=(int **)malloc(sizeof(int*)*(nnet->numLayers+1));
    float **value_upper1=(float**)malloc(sizeof(float*)*(nnet->numLayers+1));
    float **value_lower1=(float**)malloc(sizeof(float*)*(nnet->numLayers+1));
    float **symbol_upper1=(float**)malloc(sizeof(float*)*(numLayers+1));
    float **symbol_lower1=(float**)malloc(sizeof(float*)*(numLayers+1));
    for(int i=0;i<numLayers+1;i++)
    {
		value_upper1[i]=(float*)malloc(sizeof(float)*nnet->maxLayerSize);
		memcpy(value_upper1[i], value_upper[i], sizeof(float)*nnet->maxLayerSize);
		value_lower1[i]=(float*)malloc(sizeof(float)*nnet->maxLayerSize);
		memcpy(value_lower1[i], value_lower[i], sizeof(float)*nnet->maxLayerSize);
    	states1[i]=(int*)malloc(sizeof(int)*nnet->maxLayerSize);
		memcpy(states1[i], states[i], sizeof(int)*nnet->maxLayerSize);
    	states2[i]=(int*)malloc(sizeof(int)*nnet->maxLayerSize);
    	memcpy(states2[i], states[i], sizeof(int)*nnet->maxLayerSize);
    	symbol_upper1[i]=(float*)malloc(sizeof(float)*maxLayerSize*(inputSize+1));
    	memcpy(symbol_upper1[i], symbol_upper[i], sizeof(float)*maxLayerSize*(inputSize+1));
    	symbol_lower1[i]=(float*)malloc(sizeof(float)*maxLayerSize*(inputSize+1));
    	memcpy(symbol_lower1[i], symbol_lower[i], sizeof(float)*maxLayerSize*(inputSize+1));
    }


 //   printf("states1 and states2 and states is:%d,%d,%d\n",states1[3][24],states2[3][24],states[3][24]);

	lprec *lp1, *lp2;
	lp1 = copy_lp(lp);
	lp2 = copy_lp(lp);

	int rule_num1;// = *rule_num;
	int rule_num2;// = *rule_num;


//	printf("refining by split_by_predicates, case 1 begin.\n");
//	printf("c1b\n");
	memset(row, 0, Ncol*sizeof(float));

	//splitting with cons<=0
	set_add_rowmode(lp1, TRUE);
//	int flag_up_eqs_low=0;
	for(int j=1;j<Ncol+1;j++){
		row[j] = symbol_upper[split_layer][j+split_node*(inputSize+1)-1];
	}

    add_constraint(lp1, row, LE,-symbol_upper[split_layer][inputSize+split_node*(inputSize+1)]);

 //	rule_num1 += 1;

	set_add_rowmode(lp1, FALSE);
	if(property!=4)
	set_presolve(lp1,PRESOLVE_ROWS+PRESOLVE_COLS,0);
	states1[split_layer][split_node]=0;

//splitting with cons>=0
	 set_add_rowmode(lp2, TRUE);
	 add_constraint(lp2, row, GE,-symbol_upper[split_layer][inputSize+split_node*(inputSize+1)]);
//	 rule_num2 += 1;
	 set_add_rowmode(lp2, FALSE);
	if(property!=4)
	 set_presolve(lp2,PRESOLVE_ROWS+PRESOLVE_COLS,0);
	 states2[split_layer][split_node]=2;


	int unsat1=0, unsat2=0;;
    int result1=0,result2=0;
	int need_check_adv1=0,need_check_adv=0;
    int flag=0;
      if(REFINE_delay_NUM!=0){
    	if(depth % REFINE_delay_NUM==0)
		{
		//计算第i维的上下界。
			for(int j=1;j<Ncol+1;j++){
		      		row[j] = 0;
		   	}
		   set_timeout(lp2,10);
		   for(int i=0;i<inputSize;i++)
		   {
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
	     //           printf("the negation is unsat 1, and the ret is %d!\n", ret);
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
	    //            printf("the negation is unsat 2, and the ret is%d!\n", ret);
		       	    result2 = 1;
		            break;
		        } 
		        row[i+1]=0;
			if(refined==1&&unsat2==0)
			{
			//	need_check_adv=1;
		//	printf("it is refined1!\n"); 
			     if(input_interval2.lower_matrix.data[i]>input_interval2.upper_matrix.data[i]) 
				input_interval2.upper_matrix.data[i]=input_interval2.lower_matrix.data[i];
		             set_bounds(lp2,i+1,input_interval2.lower_matrix.data[i],input_interval2.upper_matrix.data[i]);
			}
		    }//end for
		}
//		if(need_check_adv==0 && result2==0)
//			printf("no refine1\n");
//		  check_adv(nnet,&input_interval2);


		//计算第i维的上下界。
		for(int j=1;j<Ncol+1;j++){
		      row[j] = 0;
		   }
		    set_timeout(lp1,10);
		    for(int i=0;i<inputSize;i++)
		    {
			int refined=0;
			row[i+1]=1;
			set_obj_fn(lp1, row);
			set_maxim(lp1);
	//       	write_LP(lp1,stdout);
			ret = solve(lp1);
	     //   	  printf("the ret is %d!\n", ret);
			if(ret == 0){
		            float temp= get_objective(lp1);
			    if(temp<input_interval1.upper_matrix.data[i])   	
	    	            {
		        	refined=1;
		  		input_interval1.upper_matrix.data[i]=temp; 
			    }	     		
	//input_interval1.upper_matrix.data[i]=get_objective(lp1);
	    //    		printf("the input_up is %f",get_objective(lp1));
			}
		        else{
	    //            printf("the negation is unsat 1, and the ret is %d!\n", ret);
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
	//       		input_interval1.lower_matrix.data[i]=get_objective(lp1);
	    //    		printf("the input_low is %f",get_objective(lp1));
			}
		        else{
	     //           printf("the negation is unsat 2, and the ret is%d!\n", ret);
		            result1 = 1;
		            break;
		        }
			if(unsat1==0 && refined==1)
		        {
		//		need_check_adv1=1;
			//	printf("it is refined!\n");      
				if(input_interval1.lower_matrix.data[i]>input_interval1.upper_matrix.data[i]) 
					input_interval1.upper_matrix.data[i]=input_interval1.lower_matrix.data[i];
		       	        set_bounds(lp1,i+1,input_interval1.lower_matrix.data[i],input_interval1.upper_matrix.data[i]);
			}
			row[i+1]=0;
		    }
	}
/*
	 pthread_mutex_lock(&lock);
    	 if(adv_found==1)
	{ pthread_mutex_unlock(&lock); return 1;}
	 pthread_mutex_unlock(&lock);
*/
	float equation_upper1[(inputSize+1)*maxLayerSize];
	float equation_lower1[(inputSize+1)*maxLayerSize];
	float new_equation_upper1[(inputSize+1)*maxLayerSize];
	float new_equation_lower1[(inputSize+1)*maxLayerSize];
/*
	memcpy(equation_upper1, equation_upper1, sizeof(float)*(inputSize+1)*maxLayerSize);
	memcpy(equation_lower1, equation_lower1, sizeof(float)*(inputSize+1)*maxLayerSize);
	memcpy(new_equation_upper1, new_equation_upper1, sizeof(float)*(inputSize+1)*maxLayerSize);
	memcpy(new_equation_lower1, new_equation_lower1, sizeof(float)*(inputSize+1)*maxLayerSize);
*/
	 pthread_mutex_lock(&lock);
//        if(0){
	 if(count<=MAX_THREAD) {
           pthread_mutex_unlock(&lock);
           pthread_t workers1, workers2;
       struct check_property_by_forward_backward_refining_args args1 = {nnet,&input_interval1,value_upper,value_lower,symbol_upper,symbol_lower,states1,\
    		   equation_upper,equation_lower,new_equation_upper,new_equation_lower,
    		   lp1,&rule_num1,property,1,depth1,split_layer1,change_split_layer1,need_check_adv1};

        struct check_property_by_forward_backward_refining_args args2 = {nnet,&input_interval2,value_upper1,value_lower1,symbol_upper1,symbol_lower1,states2,\
        		equation_upper1,equation_lower1,new_equation_upper1,new_equation_lower1,
        		lp2,&rule_num2,property,1,depth2,split_layer2,change_split_layer2,need_check_adv};
	if(result1==0)
        {
		pthread_create(&workers1, NULL, check_property_by_forward_backward_refining_thread, &args1);
	        pthread_mutex_lock(&lock);
	        count++;
	
	        pthread_mutex_unlock(&lock);
	}

	if(result2==0)
        //printf ( "pid1: %ld start %d \n", syscall(SYS_gettid), count);
        {
		pthread_create(&workers2, NULL, check_property_by_forward_backward_refining_thread, &args2);
        	pthread_mutex_lock(&lock);
        	count++;
   		pthread_mutex_unlock(&lock);
	}
        void* result21=result1;
        void* result22=result2;
        
        //printf ( "pid2: %ld start %d \n", syscall(SYS_gettid), count);
	if(result1==0)
        {	
		pthread_join(workers1, &result21);
	        pthread_mutex_lock(&lock);
        	count--;
        	pthread_mutex_unlock(&lock);
       }
	if(result2==0)
	{
        	pthread_join(workers2, &result22);
        	pthread_mutex_lock(&lock);
        	count--;
		pthread_mutex_unlock(&lock);
	}
	result1=(int)result21;
        result2=(int)result22;
       }
       else
       {
         pthread_mutex_unlock(&lock);
	
 
	
	 if(result1==0)
           result1=check_property_by_forward_backward_refining(nnet,&input_interval1,value_upper,value_lower,symbol_upper,symbol_lower,states1,\
        		   equation_upper,equation_lower,new_equation_upper,new_equation_lower,
        		   lp1,&rule_num1,property,1,depth1,split_layer1,change_split_layer1,need_check_adv1);

	  if(result2==0)
          result2 = check_property_by_forward_backward_refining(nnet,&input_interval2,value_upper1,value_lower1,symbol_upper1,symbol_lower1,states2,\
        		  equation_upper1,equation_lower1,new_equation_upper1,new_equation_lower1,
        		  lp2,&rule_num2,property,1,depth2,split_layer2,change_split_layer2,need_check_adv);
      }
    delete_lp(lp1);
    delete_lp(lp2);
        	for(int i=0;i<nnet->numLayers+1;i++)
        	{
        		free(value_upper1[i]);
        		free(value_lower1[i]);
            	free(symbol_upper1[i]);
            	free(symbol_lower1[i]);
            	free(states2[i]);
            	free(states1[i]);
        	}
        	free(value_upper1);
        	free(value_lower1);
            free(symbol_upper1);
            free(symbol_lower1);
              free(states2);
          free(states1);
//    printf("begin returning\n");
    if(result1==1&&result2==1)
    	return 1;
    else
    	return 0;
}
