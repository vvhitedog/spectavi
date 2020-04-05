import os
import re
import sys
import platform
import subprocess

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion

from setuptools.command.install import install as _install
import inspect


class CMakeExtension(Extension):

    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):

    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def clean_compile_commands_db(self,filename):
        with open(filename,'r') as f:
            lines = [ " ".join([ word for word in line.split(" ") if word != '-fopenmp' ]) for line in f.readlines()  ]
        with open(filename,'w') as f:
            for line in lines:
                f.write(line)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,]
        if os.environ.get('CMAKE_INSTALL_PREFIX') is not None:
            cmake_args.append('-DCMAKE_INSTALL_PREFIX='+os.environ['CMAKE_INSTALL_PREFIX'])

        # always make compilation db
        cmake_args.append('-DCMAKE_EXPORT_COMPILE_COMMANDS=ON')

        # XXX: if pdb is set, use a debug build
        cmdline_args = self.distribution.script_args
        cmdline_args = [arg.strip('--') for arg in cmdline_args]
        do_debug = self.debug
        if 'debug' in cmdline_args:
            do_debug = True

        # check env variables for other flags to set
        env_gperf = os.environ.get('GPERF_PROFILER_BUILD')
        env_openmp = os.environ.get('ENABLE_OPENMP')
        do_profile = env_gperf is not None and env_gperf == "ON"
        do_openmp = not(env_openmp is not None and env_openmp == "OFF")


        cfg = 'Debug' if do_debug else ( 'RelWithDebInfo' if do_profile else 'Release')
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if sys.maxsize > 2 ** 32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        if do_profile:
            cmake_args += ['-DGPERF_PROFILER_BUILD=ON']
        else:
            cmake_args += ['-DGPERF_PROFILER_BUILD=OFF']
        if do_openmp:
            cmake_args += ['-DENABLE_OPENMP=ON']
        else:
            cmake_args += ['-DENABLE_OPENMP=OFF']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        # clean up the compilation database
        self.clean_compile_commands_db(os.path.join(self.build_temp,'compile_commands.json'))
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)


def get_version_from_cmake_file(cmake_file):
    with open(cmake_file,'r') as f:
        lines = [ l.strip() for l in f.readlines() ]
    major_line = filter(lambda l: l.find('set(PROJECT_VERSION_MAJOR') != -1,lines)[0]
    minor_line = filter(lambda l: l.find('set(PROJECT_VERSION_MINOR') != -1,lines)[0]
    patch_line = filter(lambda l: l.find('set(PROJECT_VERSION_PATCH') != -1,lines)[0]
    def get_num_from_line(line):
        return (line.split(' ')[-1].split(')')[0])
    version = '.'.join(map(get_num_from_line,[major_line,minor_line,patch_line]))
    return version


__version__ = get_version_from_cmake_file('CMakeLists.txt')

setup(
    name='spectavi',
    version=__version__,
    author='Matt Gara',
    author_email='gara.matt@gmail.com',
    description='A minimalistic multi-view stereo and geometry library.',
    long_description='',
    ext_modules=[CMakeExtension('spectavi')],
    cmdclass=dict(build_ext=CMakeBuild),
    test_suite='nose.collector',
    tests_require=['nose'],
    install_requires=['numpy','cndarray'],
    packages=['spectavi'],
    zip_safe=False,
)
