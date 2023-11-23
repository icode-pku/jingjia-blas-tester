#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "helper.hh"

using testsweeper::ParamType;
using testsweeper::DataType;
using testsweeper::char2datatype;
using testsweeper::datatype2char;
using testsweeper::datatype2str;
using testsweeper::ansi_bold;
using testsweeper::ansi_red;
using testsweeper::ansi_normal;


typedef void (*test_help_func_ptr)();

typedef struct {
    const char* name;
    test_help_func_ptr func;
    int section;
} routines_help_t;


test_help_func_ptr find_help_tester(
    const char *name,
    std::vector< routines_help_t >& routines )
{
    for (size_t i = 0; i < routines.size(); ++i) {
        if (strcmp( name, routines[i].name ) == 0) {
            return routines[i].func;
        }
    }
    return nullptr;
}

enum SectionHelp {
    newline = 0,  // zero flag forces newline
    test1,
    test2,
    test3,
    aux,
    num_sections,  // last
};

std::vector< routines_help_t > routines = {
    {"example_test1", helper_test1, SectionHelp::test1},
    {"example_test2", helper_test2, SectionHelp::test1},
    {"cublassetstream", helper_cublasSetStream, SectionHelp::test1},
    {"cublasgetstream", helper_cublasGetStream, SectionHelp::test1},
    {"cublassetvector", helper_cublasSetVector, SectionHelp::test1},
    { "", nullptr, SectionHelp::newline},
};

void print_routine_name(std::vector< routines_help_t >& routines){
        for (size_t i = 0; i < routines.size(); ++i) {
            printf("%-35s",routines[i].name);
            if((i+1)%2==0) printf("\n");
    }
}

int main( int argc, char** argv )
{
    if (argc < 2
        || strcmp( argv[argc-1], "-h" ) == 0
        || strcmp( argv[argc-1], "--help" ) == 0)
    {
        printf("Please enter the following routine: \n");
        print_routine_name(routines);
        return 0;
    }
    const char* routine = argv[ argc-1 ];
    test_help_func_ptr test_help_routine = find_help_tester( routine, routines );
    if(test_help_routine ==nullptr){
        printf("input routine name error! Please enter the following routine: \n");
        print_routine_name(routines);
        return 0;
    }
    try{
        test_help_routine();
    }
    catch (const std::exception& ex) {
        fprintf( stderr, "%s%sError: %s%s\n",
                  ansi_bold, ansi_red, ex.what(), ansi_normal );
    }
    return 0;
}