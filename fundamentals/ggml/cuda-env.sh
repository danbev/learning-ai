VERSION=12.2
export PATH=/usr/local/cuda-${VERSION}/bin:$PATH                                    
export LD_LIBRARY_PATH=/usr/local/cuda-${VERSION}/lib64:$LD_LIBRARY_PATH 

# Source Spack so that we can switch to use gcc@12.1.0
. ~/work/linux/spack/share/spack/setup-env.sh
spack load gcc@12.1.0
gcc --version
