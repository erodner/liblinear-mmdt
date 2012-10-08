#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdarg.h>
#include "../linear.h"
#include "../mmdt-linear-dual.h"

#include "mex.h"
#include "linear_model_matlab.h"

#ifdef MX_API_VER
#if MX_API_VER < 0x07030000
typedef int mwIndex;
#endif
#endif

#define CMD_LEN 2048
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL

void myPrintf(const char *msg, ...)
{
   va_list argList;
   char buf[2560];
   va_start(argList, msg);
   vsprintf(buf, msg, argList);
   va_end(argList);
   mexPrintf("%s", buf);
   mexEvalString("drawnow;"); // to dump string directly.
}

void print_null(const char *s) {}
void print_string_matlab(const char *s) {myPrintf(s);}

void exit_with_help()
{
	myPrintf("[ target_model, W ] = train_linear_mmdt_fast( target_weights, target_labels, target_data, source_weights, source_labels, source_data {, options } );\n\n");
  myPrintf("- C parameters are given indirectly by instance weights\n");
  myPrintf("- Labels have to be in the range of 1..K for target and source\n");
  myPrintf("- options is a MATLAB struct including optimization parameters (eps, calculate_hinge, explicitW, max_iterations)\n");
  myPrintf("\nLiblinear-MMDT Solver by Erik Rodner (2012)\n");
}

static void fake_answer(mxArray *plhs[])
{
	plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
	plhs[1] = mxCreateDoubleMatrix(0, 0, mxREAL);
}


problem *read_problem_sparse(const mxArray *weight_vec, const mxArray *label_vec, const mxArray *instance_mat, struct feature_node * & x_space, int & max_class)
{
	set_print_string_function(print_string_matlab);

  problem *prob = Malloc(problem,1);


	int i, j, k, low, high;
	mwIndex *ir, *jc;
	int elements, max_index, num_samples, label_vector_row_num, weight_vector_row_num;
	double *samples, *labels, *weights;

  // instance sparse matrix in column format
	mxArray *instance_mat_col; 
  // for simplicity for only allow for column format
	instance_mat_col = (mxArray *)instance_mat;

  // some simple initialization
	prob->x = NULL;
	prob->y = NULL;
	prob->W = NULL;
	x_space = NULL;


	// get the number of instance
	prob->l = (int) mxGetN(instance_mat_col);
  //myPrintf("[ReadProblemSparse] number of examples = %d\n", prob->l);

  
  // check the length of the weight and label vector
	weight_vector_row_num = (int) mxGetM(weight_vec);
	label_vector_row_num = (int) mxGetM(label_vec);

	if(weight_vector_row_num == 0) 
		myPrintf("Warning: treat each instance with weight 1.0\n");
	else if(weight_vector_row_num!=prob->l)
	{
		myPrintf("Length of weight vector does not match # of instances.\n");
		return NULL;
	}

	if(label_vector_row_num!=prob->l)
	{
		myPrintf("Length of label vector does not match # of instances.\n");
		return NULL;
	}
	
	// each column is one instance
	weights = mxGetPr(weight_vec);
	labels = mxGetPr(label_vec);
	samples = mxGetPr(instance_mat_col);

  // allocate memory
	ir = mxGetIr(instance_mat_col);
	jc = mxGetJc(instance_mat_col);

	num_samples = (int) mxGetNzmax(instance_mat_col);

	elements = num_samples + prob->l*2;
	max_index = (int) mxGetM(instance_mat_col);

	prob->y = Malloc(double, prob->l);
	prob->W = Malloc(double, prob->l);
	prob->x = Malloc(struct feature_node*, prob->l);
	x_space = Malloc(struct feature_node, elements);

  // we also expect that the bias is already incorporated, so that
  // we don't have to take care about it
	prob->bias = -1.0;

  // read sparse elements
  max_class = 0;
	j = 0;
	for(i=0;i<prob->l;i++)
	{
		prob->x[i] = &x_space[j];
		prob->y[i] = labels[i];

    if ( (int)prob->y[i] > max_class )
      max_class = (int)prob->y[i];

		prob->W[i] = 1;
		if(weight_vector_row_num == prob->l)
			prob->W[i] *= (double) weights[i];
		low = (int) jc[i], high = (int) jc[i+1];
		for(k=low;k<high;k++)
		{
			x_space[j].index = (int) ir[k]+1;
			x_space[j].value = samples[k];
			j++;
	 	}
		x_space[j++].index = -1;
	}

	prob->n = max_index;

	return prob;
}

// TODO
//  - better convergence checks
//  - use initial W and theta estimates for the solvers
//
//
// Interface function of matlab -------------------------------------
//
// prhs[0]: target weights
// prhs[1]: target labels (1..K)
// prhs[2]: target examples (sparse matrix)
// prhs[3]: source weights
// prhs[4]: source labels (1..K)
// prhs[5]: source examples (sparse matrix)
void mexFunction( int nlhs, mxArray *plhs[],
		int nrhs, const mxArray *prhs[] )
{
  problem *target_prob = NULL;
  problem *source_prob = NULL;

  // we can skip this maybe
  struct parameter param;
  struct mmdt_parameter mmdt_param;

  // only this solver really matters!
  param.solver_type = L2R_L2LOSS_SVC_DUAL;

  // C parameter is incorporated by the weights, therefore, we use a standard setting here
	param.C = 1;
	param.eps = 0.1; 
	param.p = 0.1;
	param.nr_weight = 0;
  // the following settings are only class-wise weights
	param.weight_label = NULL;
	param.weight = NULL;

	// fix random seed to have same results for each run
	// (for cross validation)
	srand(1);

	// Transform the input Matrix to libsvm format
	if( nrhs >= 6 )
	{
		if(!mxIsDouble(prhs[0]) || 
       !mxIsDouble(prhs[1]) || 
       !mxIsDouble(prhs[2]) || 
       !mxIsDouble(prhs[3]) || 
       !mxIsDouble(prhs[4]) || 
       !mxIsDouble(prhs[5])) 
    {
			myPrintf("Error: weight and label vectors as well as data matrices must be double\n");
			fake_answer(plhs);
			return;
		}
    
    // first, let us use default values for the options, which
    // can of course be overwritten 
    std_mmdt_parameter ( &mmdt_param );

    // check whether some options are given
    if ( nrhs == 7 )
    {
      // check whether we really have a MATLAB struct
      if ( !mxIsStruct(prhs[6]) )
      {
        myPrintf("Error: last argument has to be a MATLAB struct\n");
        fake_answer(plhs);
        return;
      }
      const mxArray *options = prhs[6];

      // temporary pointer to MATLAB data associated with a key
      // within the struct
      mxArray *value;
      
      value = mxGetField ( options, 0, "max_iterations" );
      if ( value != NULL ) {
        mmdt_param.max_iterations = (int) ((mxGetPr(value))[0]);
        myPrintf("[MMDT Parameter] max iterations set to %d\n", mmdt_param.max_iterations );
      }
        
      value = mxGetField ( options, 0, "calculate_hinge" );
      if ( value != NULL )
      {
        mmdt_param.calculate_hinge = ((mxGetPr(value))[0] > 0) ? true : false;

        myPrintf("[MMDT Parameter] calculating the hinge loss and objective? %s\n", mmdt_param.calculate_hinge ? "yes" : "no");
      }

      value = mxGetField ( options, 0, "explicitW" );
      if ( value != NULL )
      {
        mmdt_param.explicitW = ((mxGetPr(value))[0] > 0) ? true : false;
        myPrintf("[MMDT Parameter] explicit W transformation? %s\n", mmdt_param.explicitW ? "yes" : "no");
      }

      value = mxGetField ( options, 0, "warm_start_W" );
      if ( value != NULL )
      {
        mmdt_param.warm_start_W = ((mxGetPr(value))[0] > 0) ? true : false;
        myPrintf("[MMDT Parameter] warm start of W? %s\n", mmdt_param.warm_start_W ? "yes" : "no");
      }

      value = mxGetField ( options, 0, "regularize_identity" );
      if ( value != NULL )
      {
        mmdt_param.regularize_identity = ((mxGetPr(value))[0] > 0) ? true : false;
        myPrintf("[MMDT Parameter] regularize W with respect to identity matrix? %s\n", mmdt_param.regularize_identity ? "yes" : "no");
      }

      value = mxGetField ( options, 0, "warm_start_theta" );
      if ( value != NULL )
      {
        mmdt_param.warm_start_theta = ((mxGetPr(value))[0] > 0) ? true : false;
        myPrintf("[MMDT Parameter] warm start of theta? %s\n", mmdt_param.warm_start_theta ? "yes" : "no");
      }

      value = mxGetField ( options, 0, "verbose_level" );
      if ( value != NULL )
        mmdt_param.verbose_level = (int)((mxGetPr(value))[0]);

      value = mxGetField ( options, 0, "eps" );
      if ( value != NULL )
      {
        mmdt_param.eps = (double)((mxGetPr(value))[0]);
        myPrintf("[MMDT Parameter] eps set to %f\n", mmdt_param.eps );
      }
      
      value = mxGetField ( options, 0, "epsW" );
      if ( value != NULL )
      {
        mmdt_param.epsW = (double)((mxGetPr(value))[0]);
        myPrintf("[MMDT Parameter] epsW set to %f\n", mmdt_param.epsW );
      }

      value = mxGetField ( options, 0, "max_iter" );
      if ( value != NULL )
      {
        mmdt_param.max_iter = (int)((mxGetPr(value))[0]);
        myPrintf("[MMDT Parameter] max_iter set to %d\n", mmdt_param.max_iter );
      }

      value = mxGetField ( options, 0, "ratio_active_size" );
      if ( value != NULL )
      {
        mmdt_param.ratio_active_size = (float)((mxGetPr(value))[0]);
        myPrintf("[MMDT Parameter] ratio_active_size set to %f\n", mmdt_param.ratio_active_size );
      }


    }

    feature_node *source_x_space;
    feature_node *target_x_space;
    int nr_class;
		if(mxIsSparse(prhs[2]) && mxIsSparse(prhs[5]))
    {
      int max_class_source;
      int max_class_target;
			target_prob = read_problem_sparse(prhs[0], prhs[1], prhs[2], source_x_space, max_class_source);
			source_prob = read_problem_sparse(prhs[3], prhs[4], prhs[5], target_x_space, max_class_target);

      // some error occurred during MATLAB -> C conversion
      if ( (target_prob == NULL) || (source_prob == NULL) )
      {
        destroy_param(&param);
			  fake_answer(plhs);
        return;
      }

      //if ( max_class_target != max_class_source )
      //{
      //  myPrintf("Label spaces seem to differ, which is not yet supported.\n");
      //  destroy_param(&param);
			//  fake_answer(plhs);
      //}

      if ( max_class_source == 1 )
      {
        myPrintf("\n\nTwo classes are not really supported! Take care :)\n\n\n");
        // weird hack, sorry!
        // the problem will be still treated with K = 2
        // and labels are assumed to be 1..2
        nr_class = 2;
      } else {
        nr_class = max_class_source;
      }
		} else {
			myPrintf("Data matrices must be sparse; use sparse() first\n");
			destroy_param(&param);
			fake_answer(plhs);
			return;
		}


    /*
    myPrintf("[MMDT Solver] source dimension is %d\n", source_prob->n );
    myPrintf("[MMDT Solver] target dimension is %d\n", target_prob->n );
    myPrintf("[MMDT Solver] number of tasks %d\n", nr_class );
    */

    // allocate memory for the transformation matrix
    plhs[1] = mxCreateDoubleMatrix( source_prob->n, target_prob->n, mxREAL );
    double *w_transform = mxGetPr(plhs[1]);
   
  
    // run the solver
    model *hyperplanes_model = mmdt_solver ( target_prob, source_prob, w_transform, &param, &mmdt_param );

   for ( int i = 0; i < 10 ; i++ )
      fprintf(stderr, "W Matrix(%d,%d) %f\n", i, i, w_transform[i*target_prob->n + i] );
    // transform w_transform and w_hyperplanes to MATLAB matrices and return arguments
    // w_transform already uses a return pointer and MATLAB matrix
  	model_to_matlab_structure(plhs, hyperplanes_model);

    free(source_x_space);
    free(target_x_space);
    // the data instances are not valid anymore!
  } else {
    exit_with_help();
  }

  destroy_param(&param);

  if ( source_prob != NULL )
  {
    free(source_prob->y);
    free(source_prob->x);
    free(source_prob->W);
    free(source_prob);  
  }

  if ( target_prob != NULL )
  {
    free(target_prob->y);
    free(target_prob->x);
    free(target_prob->W);
    free(target_prob);
  } 
}
