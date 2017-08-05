#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include "svm.h"
#include<iostream>

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

struct svm_model *model;
struct svm_node *x;
int max_nr_attr = 64;
static int (*info)(const char *fmt,...) = &printf;
static char *line = NULL;
static int max_line_len;

void predict(FILE *input, FILE *output);
//读取输入文件的一行
static char* readline(FILE *input)
{
	int len;
	
	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}

//输入错误退出
void exit_input_error(int line_num)
{
	fprintf(stderr,"Wrong input format at line %d\n", line_num);
	exit(1);
}

int main( )
{
	char *input_file_name = "E:\\项目程序\\vs\\神经网络\\Predict2\\trainFile.txt";
	char *model_file_name = "E:\\项目程序\\vs\\神经网络\\Predict2\\model.txt";
	char *outputname = "E:\\项目程序\\vs\\神经网络\\Predict2\\1.txt";

	FILE *input=NULL;
	fopen_s(&input,input_file_name,"r");
	FILE *output =NULL;
	fopen_s(&output,outputname,"w");

	model = svm_load_model(model_file_name);

	x = (struct svm_node *) malloc(max_nr_attr*sizeof(struct svm_node));	
	predict(input,output);	
	
	svm_free_and_destroy_model(&model);
	free(x);
	free(line);
	fclose(input);	
	fclose(output);
	return 0;
}
	
void predict(FILE *input, FILE *output)
{
	int total = 0;
	int svm_type=svm_get_svm_type(model);
	int nr_class=svm_get_nr_class(model);
  	double *prob_estimates=NULL;   

	max_line_len = 1024;
	line = (char *)malloc(max_line_len*sizeof(char));
	while(readline(input) != NULL)
	{
		int i = 0;
		double  predict_label;
		char *idx, *val, *label, *endptr;
		int inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0

		label=strtok(line," \t\n");
		
		if(label == NULL) // empty line
			exit_input_error(total+1);
		
		strtod(label,&endptr);
		if(endptr == label || *endptr != '\0')
			exit_input_error(total+1);
	

		while(1)
		{
			if(i>=max_nr_attr-1)	// need one more for index = -1
			{
				max_nr_attr *= 2;
				x = (struct svm_node *) realloc(x,max_nr_attr*sizeof(struct svm_node));
			}

			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)           
				break;
			errno = 0;
     		x[i].index = (int) strtol(idx,&endptr,10);
			if(endptr == idx || errno != 0 || *endptr != '\0' || x[i].index <= inst_max_index)
				exit_input_error(total+1);
			else
				inst_max_index = x[i].index;    

			errno = 0;
			x[i].value = strtod(val,&endptr);
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(total+1);

			++i;
		}
		x[i].index = -1;
		std::cout<<"i = "<<i<<std::endl;
		for(int i = 0; i <= 3; i++)
		{
			std::cout<<x[i].index<<"\t"<<x[i].value<<std::endl;
		}
		predict_label = svm_predict(model,x);
		fprintf(output,"%g\n",predict_label);			
	}		
}