# See the README for description of how to define these.
# export GradDescFolder=~/einkornlite/dist-newstyle/build/x86_64-osx/ghc-8.10.4/einkorn-lite-0.1.0.0/x/GradDesc.so/build/GradDesc.so
export EinkornLiteCabalFolder=~/einkornlite/dist-newstyle/build/x86_64-osx/ghc-8.10.4/einkorn-lite-0.1.0.0

export PathToGHCUP=~
export GHCVersion=8.10.4

export GHCFolder=$PathToGHCUP/.ghcup/ghc/$GHCVersion/lib/ghc-$GHCVersion/


# DO NOT EDIT ANYTHING BELOW THIS COMMENT!!
# Everything should run properly, assuming that GradDescFolder and GHCFolder are correctly set.

# Create new variable file paths from user-defined ones
export GradDescFolder=$EinkornLiteCabalFolder/x/GradDesc.so/build/GradDesc.so

# Compile exportInC.c to a dynamic shared library named out.
gcc -o out \
  exportInC.c \
  $GradDescFolder/GradDesc.so \
  -I/$GHCFolder/include \
  -I/$EinkornLiteCabalFolder/build/EinkornLite/GradDesc/ \
  -L/$GHCFolder/rts \
  -lHSrts-ghc8.10.4 \
  -O2 -fPIC -dynamic

# Run the shared dynamic library code, using 4 inputs (specified in Python, user shouldn't worry about the ordering)
export LD_LIBRARY_PATH=$GradDescFolder/:$GHCFolder/rts
./out $1 $2 $3 $4