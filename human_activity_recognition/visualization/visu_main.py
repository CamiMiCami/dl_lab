"""
This script was developed by Shuaike Liu
Order of execution:
1. visualization_true.py
2. predict.py
3. visualization_predict.py


=======================================================================================================================

predict.py and visualization_predict.py need to be executed seperately, else:

I tensorflow/core/platform/cpu_feature_guard.cc:193]
This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN)
to use the following CPU instructions in performance-critical operations:  AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
OMP: Error #15: Initializing libiomp5, but found libiomp5md.dll already initialized.
OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program. 
That is dangerous, since it can degrade performance or cause incorrect results. 
The best thing to do is to ensure that only a single OpenMP runtime is linked into the process, 
e.g. by avoiding static linking of the OpenMP runtime in any library. As an unsafe, unsupported, 
undocumented workaround you can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE
to allow the program to continue to execute, but that may cause crashes or silently produce incorrect results.
For more information, please see http://openmp.llvm.org/
"""
