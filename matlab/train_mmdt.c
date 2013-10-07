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

   mexEvalString("drawnow;"); // to dump string.
}

void print_null(const char *s) {}
void print_string_matlab(const char *s) {myPrintf(s);}

void exit_with_help()
{
	myPrintf(
	"Usage: mmdt_model = train_mmdt(hyperplane_model, weight_vector, training_label_vector, training_instance_matrix, 'liblinear_options', 'col');\n");
}

// liblinear arguments
struct parameter param;		// set by parse_command_line
struct problem prob;		// set by read_problem
struct model *model_;
struct model hyperplane_model;
struct feature_node *x_space;
int cross_validation_flag;
int col_format_flag;
int nr_fold;
double bias;


// nrhs should be 4
int parse_command_line(int nrhs, const mxArray *prhs[], char *model_file_name)
{
	int i, argc = 1;
	char cmd[CMD_LEN];
	char *argv[CMD_LEN/2];
	void (*print_func)(const char *) = print_string_matlab;	// default printing to matlab display

	// default values
	param.solver_type = L2R_L2LOSS_SVC_DUAL;
	param.C = 1;
	param.eps = INF; // see setting below
	param.p = 0.1;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;
	cross_validation_flag = 0;
	col_format_flag = 0;
	bias = -1;


	if(nrhs <= 3)
		return 1;

	if(nrhs == 6)
	{
		mxGetString(prhs[5], cmd, mxGetN(prhs[5])+1);
		if(strcmp(cmd, "col") == 0)
			col_format_flag = 1;
	}

	// put options in argv[]
	if(nrhs > 4)
	{
		mxGetString(prhs[4], cmd,  mxGetN(prhs[4]) + 1);
		if((argv[argc] = strtok(cmd, " ")) != NULL)
			while((argv[++argc] = strtok(NULL, " ")) != NULL)
				;
	}

	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		++i;
		if(i>=argc && argv[i-1][1] != 'q') // since option -q has no parameter
			return 1;
		switch(argv[i-1][1])
		{
			case 's':
				param.solver_type = atoi(argv[i]);
				break;
			case 'c':
				param.C = atof(argv[i]);
				break;
			case 'p':
				param.p = atof(argv[i]);
				break;
			case 'e':
				param.eps = atof(argv[i]);
				break;
			case 'B':
				bias = atof(argv[i]);
				break;
			case 'v':
				cross_validation_flag = 1;
				nr_fold = atoi(argv[i]);
				if(nr_fold < 2)
				{
					myPrintf("n-fold cross validation: n must >= 2\n");
					return 1;
				}
				break;
			case 'w':
				++param.nr_weight;
				param.weight_label = (int *) realloc(param.weight_label,sizeof(int)*param.nr_weight);
				param.weight = (double *) realloc(param.weight,sizeof(double)*param.nr_weight);
				param.weight_label[param.nr_weight-1] = atoi(&argv[i-1][2]);
				param.weight[param.nr_weight-1] = atof(argv[i]);
				break;
			case 'q':
				print_func = &print_null;
				i--;
				break;
			default:
				myPrintf("unknown option\n");
				return 1;
		}
	}

	set_print_string_function(print_func);

	if(param.eps == INF)
	{
		switch(param.solver_type)
		{
			case L2R_LR: 
			case L2R_L2LOSS_SVC:
				param.eps = 0.01;
				break;
			case L2R_L2LOSS_SVR:
				param.eps = 0.001;
				break;
			case L2R_L2LOSS_SVC_DUAL: 
			case L2R_L1LOSS_SVC_DUAL: 
			case MCSVM_CS: 
			case L2R_LR_DUAL: 
				param.eps = 0.1;
				break;
			case L1R_L2LOSS_SVC: 
			case L1R_LR:
				param.eps = 0.01;
				break;
			case L2R_L1LOSS_SVR_DUAL:
			case L2R_L2LOSS_SVR_DUAL:
				param.eps = 0.1;
				break;
		}
	}
	return 0;
}

static void fake_answer(mxArray *plhs[])
{
	plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
}

int read_problem_sparse(const mxArray *weight_vec, const mxArray *label_vec, const mxArray *instance_mat)
{
	int i, j, k, low, high;
	mwIndex *ir, *jc;
	int elements, max_index, num_samples, label_vector_row_num, weight_vector_row_num;
	double *samples, *labels, *weights;
	mxArray *instance_mat_col; // instance sparse matrix in column format

	prob.x = NULL;
	prob.y = NULL;
	prob.W = NULL;
	x_space = NULL;

	if(col_format_flag)
		instance_mat_col = (mxArray *)instance_mat;
	else
	{
    myPrintf("Transposing matrix ...\n");
		// transpose instance matrix
		mxArray *prhs[1], *plhs[1];
		prhs[0] = mxDuplicateArray(instance_mat);
		if(mexCallMATLAB(1, plhs, 1, prhs, "transpose"))
		{
			myPrintf("Error: cannot transpose training instance matrix\n");
			return -1;
		}
		instance_mat_col = plhs[0];
		mxDestroyArray(prhs[0]);
	}

	// the number of instance
	prob.l = (int) mxGetN(instance_mat_col);
  myPrintf("prob.l = %d\n", prob.l);
	weight_vector_row_num = (int) mxGetM(weight_vec);
	label_vector_row_num = (int) mxGetM(label_vec);

	if(weight_vector_row_num == 0) 
		myPrintf("Warning: treat each instance with weight 1.0\n");
	else if(weight_vector_row_num!=prob.l)
	{
		myPrintf("Length of weight vector does not match # of instances.\n");
		return -1;
	}
	if(label_vector_row_num!=prob.l)
	{
		myPrintf("Length of label vector does not match # of instances.\n");
		return -1;
	}
	
	// each column is one instance
	weights = mxGetPr(weight_vec);
	labels = mxGetPr(label_vec);
	samples = mxGetPr(instance_mat_col);
	ir = mxGetIr(instance_mat_col);
	jc = mxGetJc(instance_mat_col);

	num_samples = (int) mxGetNzmax(instance_mat_col);

	elements = num_samples + prob.l*2;
	max_index = (int) mxGetM(instance_mat_col);

	prob.y = Malloc(double, prob.l);
	prob.W = Malloc(double,prob.l);
	prob.x = Malloc(struct feature_node*, prob.l);
	x_space = Malloc(struct feature_node, elements);

	prob.bias=bias;

	j = 0;
	for(i=0;i<prob.l;i++)
	{
		prob.x[i] = &x_space[j];
		prob.y[i] = labels[i];
		prob.W[i] = 1;
		if(weight_vector_row_num == prob.l)
			prob.W[i] *= (double) weights[i];
		low = (int) jc[i], high = (int) jc[i+1];
		for(k=low;k<high;k++)
		{
			x_space[j].index = (int) ir[k]+1;
			x_space[j].value = samples[k];
			j++;
	 	}
		if(prob.bias>=0)
		{
			x_space[j].index = max_index+1;
			x_space[j].value = prob.bias;
			j++;
		}
		x_space[j++].index = -1;
	}

	if(prob.bias>=0)
		prob.n = max_index+1;
	else
		prob.n = max_index;

	return 0;
}

// Interface function of matlab
// now assume prhs[0]: hyperplane model, prhs[1]: label prhs[2]: features
void mexFunction( int nlhs, mxArray *plhs[],
		int nrhs, const mxArray *prhs[] )
{
	const char *error_msg;
	// fix random seed to have same results for each run
	// (for cross validation)
	srand(1);

	// Transform the input Matrix to libsvm format
	if(nrhs > 3 && nrhs < 7)
	{
		int err=0;

		if(!mxIsStruct(prhs[0]) || !mxIsDouble(prhs[1]) || !mxIsDouble(prhs[2]) || !mxIsDouble(prhs[3])) {
			myPrintf("Error: first argument should be struct and weight vector, label vector and instance matrix must be double\n");
			fake_answer(plhs);
			return;
		}

    matlab_matrix_to_model (&hyperplane_model, prhs[0]);

		if(parse_command_line(nrhs, prhs, NULL))
		{
			exit_with_help();
			destroy_param(&param);
			fake_answer(plhs);
			return;
		}

		if(mxIsSparse(prhs[3]))
			err = read_problem_sparse(prhs[1], prhs[2], prhs[3]);
		else
		{
			myPrintf("Training_instance_matrix must be sparse; "
				"use sparse(Training_instance_matrix) first\n");
			destroy_param(&param);
			fake_answer(plhs);
			return;
		}

		// train's original code
		error_msg = check_parameter(&prob, &param);

		if(err || error_msg)
		{
			if (error_msg != NULL)
				myPrintf("Error: %s\n", error_msg);
			destroy_param(&param);
			free(prob.y);
			free(prob.x);
			free(x_space);
			fake_answer(plhs);
			return;
		}

		/*if(cross_validation_flag)
		{
			double *ptr;
			plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
			ptr = mxGetPr(plhs[0]);
			ptr[0] = do_cross_validation();
		}
		else */
		{
			const char *error_msg;
			//model_ = train(&prob, &param);
	    model_ = Malloc(model,1);
      model_->bias = prob.bias;
      model_->nr_class = 2; // all tasks are considered jointly
      int dim_t = prob.n;
      int dim_s = hyperplane_model.nr_feature;
      int nr_class = hyperplane_model.nr_class;
      int nt = prob.l;
      myPrintf("source dimension is %d\n", dim_s );
      myPrintf("target dimension is %d\n", dim_t );
    
      model_->nr_feature = dim_s * dim_t;
      myPrintf("total dimension of the transformation matrix is %d\n", model_->nr_feature );
  	

      model_->w = Malloc ( double, model_->nr_feature );
      model_->label = NULL;
      model_->param = param;

      struct mmdt_model transform_model;
      initialize_mmdt_model ( &transform_model, nr_class, dim_t, dim_s, nt, true, false );
      transform_model.w_transform = &model_->w[0];

      solve_l2r_l1l2_mmdt( &prob, &hyperplane_model, &transform_model, param.eps, param.solver_type, true, 1000, 1.0 );
     
    
			error_msg = model_to_matlab_structure(plhs, model_);
			if(error_msg)
				myPrintf("Error: can't convert libsvm model to matrix structure: %s\n", error_msg);
      
      transform_model.w_transform = NULL;
      free_mmdt_model ( &transform_model );

			free_and_destroy_model(&model_);
		}
		destroy_param(&param);
		free(prob.y);
		free(prob.x);
		free(prob.W);
		free(x_space);
	}
	else
	{
		exit_with_help();
		fake_answer(plhs);
		return;
	}
}
