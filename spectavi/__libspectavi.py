import ctypes as ct
import os

"""
Get library path and load.
"""
egg_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
lib_path = os.path.join(egg_path, 'libspectavi.so')
clib = ct.cdll.LoadLibrary(lib_path)
