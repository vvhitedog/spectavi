# Spectavi

Multi-view geometry and multi-view stereo problems are fundamental problems in
optical 3D reconstruction. Modern implementations tend to obscure the
fundamental models that they rely on in the complexity of the algorithms that
are used to do the necessary reconstruction tasks.

Spectavi's purpose is to elucidate those very models in a minimalistic way. It
is largely a labor of love and learning, at least at this point. As of writing
this today, Spectavi is not meant to be competitive with more complete MVG or
MVS packages.

## Design Philosophy

Spectavi is a largely python project with a C++ backend to make certain tasks
more managable for high-resolution datasets.

The library has both a suite of tests and examples.

There are unit-tests that are designed to test that the algorithms
implemented functionally produce the correct output (usually based
on randomly generated data.) In this sense, the tests can be instructive in
showing how individual subroutines can be useful on their own.

The examples can be seen as integration tests where several subroutines may
work together to produce a sensible output from (typically) *real* data. The
examples are structured in a way that modularly show individual tasks that must
be solved in either the MVG or MVS problems, and can be chained together
sequentially to produce a useful product using a series of steps.

## Self Contained

This library is meant to be as self-contained as possible. However, some
algorithms that give good results (ex. SIFT or approximate NN) are complex
enough that they fall outside the scope of the purpose of this project (which
is to elucidate,) but would be folly to overlook. In cases like this, external
projects which have high-quality implementations are linked as dependencies of
Spectavi through git-submodules. 

The only singular dependency that does not fit this description is the
[ctypes_ndarray](https://github.com/vvhitedog/ctypes_ndarray) project, which is
a simple implementation of interfacing numpy data with C++ (and vice-versa),
which Spectavi adopts into its methodology.

## Feature Inventory and Roadmap

### Multi View Geometry
- [x] Two-view geometry Essential Matrix estimation: implementation
- [x] Two-view geometry Essential Matrix estimation: testing
- [ ] Two-view geometry Essential Matrix estimation: example (real-data)
- [x] Two-view geometry point triangulation (DLT Direct Linear Transform): implementation
- [x] Two-view geometry point triangulation (DLT Direct Linear Transform): testing
- [ ] Two-view geometry point triangulation (DLT Direct Linear Transform): example (real-data)
- [x] Seven point algorithm (Estimating Essential Matrix): implementation
- [x] Seven point algorithm (Estimating Essential Matrix): testing
- [x] Two-view geometry image rectification given an Essential Matrix: implementation
- [ ] Two-view geometry image rectification given an Essential Matrix: example

### Feature
- [ ] SIFT feature detection and descriptor (vlfeat): implementation
- [ ] SIFT feature detection and descriptor: testing
- [ ] SIFT feature detection and descriptor: example
- [ ] ANN using SVD: implementation
- [ ] ANN using [hnswlib](https://github.com/nmslib/hnswlib): implementation
- [ ] ANN check against real computation: testing
- [ ] ANN ratio-test: example

Notes:
- Some algorithms (such as the Seven point algorithm) there is not much point showing it's use on real-data as it would be extremely limited.
- Some algorithms (such as image rectification) it is easier to verify it works on real-data rather than invent a contrived example for unit-testing.
- ANN := Approximate Nearest Neighbour

## Install

Spectavi has been desinged in Python 2.7, and may not work with newer versions.
The only python dependecies are numpy and cndarray. Installation of numpy
is fairly standard and is not described here (since there are many ways to do
it, each with their own pros and cons.)

1. Clone the repo: `git clone https://github.com/vvhitedog/spectavi`
2. Init and update all submodules inside the repo: `git submodule update --init --recursive`
3. Install `cndarray` for user (for global install remove `--user` and run as root or in a virtual env): `pushd ctypes_ndarray && python setup.py install --user && popd` 
4. Install `spectavi` for user (for global install remove `--user` and run as root or in a virtual env): `python setup.py install --user`

## Build

To just build Spectavi, the instructions are almost the same as install, except instead of step 4., replace with 

4. Build `spectavi`: `python setup.py build`


## Tests

Spectavi uses nose tests and python's unittest to do unit testing. To run unit
tests, make sure that you can successfully build Spectavi, then use

* Run `python setup.py  nosetests  --nocapture` to see print statements.
* Run `python setup.py  nosetests` to not see print statements.


## Debugging

To enable debugging in any context, Spectavi can be built with debugging enabled for the C++ backend.

* Build `spectavi` with debug: `python setup.py build --debug`
* Run `gdb -arg python setup.py  nosetests --debug ` to run Spectavi unit tests with C++ code compiled as debug under `gdb`.

