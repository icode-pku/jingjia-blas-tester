#ifndef HELPER_HH
#define HELPER_HH


#include "testsweeper.hh"
#include "blas.hh"

using llong = long long;

#include "cblas_wrappers.hh"
#include "lapack_wrappers.hh"
#include "blas/flops.hh"
#include "print_matrix.hh"
#include "check_gemm.hh"
#include  "../src/device_internal.hh"


void helper_cublasSetStream();
void helper_cublasGetStream();
void helper_cublasSetVector();
void helper_cublasGetVector();
void helper_cublasSetVectorAsync();
void helper_cublasGetVectorAsync();
void helper_cublasSetPointerMode();
void helper_cublasGetPointerMode();
void helper_cublasLoggerConfigure();
void helper_cublasGetLoggerCallback();
void helper_cublasSetLoggerCallback();

void helper_test1();
void helper_test2();


inline
bool check_return_status(cublasStatus_t device_status, const char* excpet_status, int &all_tests, int &passed_tests, int &failed_test)
{
    const char* device_status_name = blas::device_errorstatus_to_string(device_status);
    all_tests++;
    if(strcmp(device_status_name, excpet_status)==0){
        passed_tests++;
        return true;
    }
    //printf("API %s, Error status: %s Except error status: %s\n",api_name, device_status_name, excpet_status);
    failed_test++;
    return false;
}

class TestId{
    public:
        int test_id;
        int all_testcase;
        std::string param_name;
        std::vector<std::string> params_name;
    public:
        TestId(){}
        TestId(int all_testcase_, std::string param_name_){
            all_testcase = all_testcase_;
            param_name = param_name_;
            test_id = 0;
            param_parse();
        }
        TestId(int all_testcase_, std::string param_name_, int id){
            all_testcase = all_testcase_;
            param_name = param_name_;
            test_id = id;
            param_parse();
        }

        void TestProblemHeader(int param_id, bool flag=false, const char* inputvalue = nullptr){
            test_id++;
            if(flag) printf("[%d/%d] When all parameters are legal\n",test_id, all_testcase);
            else printf("[%d/%d] When params %d \"%s\" is \"%s\"\n",test_id, all_testcase, param_id, params_name[param_id].c_str(), inputvalue);
        }

        void TestApiHeader(){
            int len = param_name.length();
            int pre=0, param_id=0;
            for(int i=0; i<len; i++)
            {
                if(param_name[i]=='('){
                    printf("We will test the %s api, its parameters are:\n", (param_name.substr(pre, i-pre)).c_str());
                    pre = i+1;break;
                }
                // if(params[i]==' '){ pre=i+1;}
                // if(params[i]==','|| params[i]==')'){
                //     printf("param %d: %s  ", param_id++, (s.substr(pre, i-pre)).c_str());
                //     pre=i+1;
                // }
            }
            printf("%s\n", param_name.c_str());
        }

        void param_parse(){
            int len = param_name.length();
            params_name.clear();
            int pre=0;
            for(int i=0; i<len; i++){
                if(param_name[i]=='('){pre=i+1;}
                if(param_name[i]==' '){pre=i+1;}
                if(param_name[i]==','|| param_name[i]==')'){
                    params_name.push_back(param_name.substr(pre, i-pre));
                    pre=i+1;
                }
            }
        }
};


#endif