from distutils.core import setup, Extension
from Cython.Build import cythonize

COMPILE_ARGS = ['-g', '-std=c++11', "-stdlib=libc++", "-I/opt/homebrew/Cellar/boost/1.82.0_1/include/", "-I/opt/homebrew/Cellar/llvm/17.0.2/include/", '-mmacosx-version-min=10.9'] #, '-std=c++11']

setup(ext_modules = cythonize(Extension(
           "cpp_mstar",                                
           sources=["cython_od_mstar.pyx"], 
           extra_compile_args=["-std=c++11", "-stdlib=libc++", "-I/opt/homebrew/Cellar/boost/1.82.0_1/include/", '-mmacosx-version-min=10.9' ],
          #  extra_link_args=["-L/usr/local/opt/llvm/lib/"]
          #  extra_link_args=["-L/opt/homebrew/opt/llvm/lib/"]
      )))