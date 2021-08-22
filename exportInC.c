#include <stdio.h>
#include "HsFFI.h"
// #include "/Users/nathanpowell/einkornlite/dist-newstyle/build/x86_64-osx/ghc-8.10.4/einkorn-lite-0.1.0.0/build/EinkornLite/GradDesc/Export_stub.h"
#include "Export_stub.h"

// #include <cstddef.h>

// void init_hs(int argc, char **argv) {
//        hs_init(&argc, &argv);

// void init_hs(){
//     int v = 0;
//     hs_init(&v, NULL);  
// }; 

// void useCompileC(char c_ein_formula_fname[], char c_filename[], char c_desc_str[]) {
//     compileC(c_ein_formula_fname, c_filename, c_desc_str);
// }

// void exit_hs() {
//     hs_exit();
// }

// char* useProcess(char *inputs, char *loss, char *src, char *outputFileName){
//     // printf("%s\n", *inputs);
//     // init_hs();
//     char *childDependencies = process(inputs, loss, src, outputFileName);
//     // hs_exit();
//     return childDependencies;
// }

int main(int argc, char *argv[])
{
    // Must initialize Haskell and the arguments before passing into shared Haskell code.
    hs_init(&argc, &argv);

    // Pass the 4 arguments into the process method in export.hs in src/Einkorn/gradDesc
    process(argv[1], argv[2], argv[3], argv[4]);

    // Must exit after finishing use of Haskell code.
    hs_exit();

    return 0;
}