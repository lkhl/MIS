#include <random>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>


typedef Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> IntMatrix;
typedef Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> FloatMatrix;
typedef Eigen::Array<int, 1, Eigen::Dynamic, Eigen::RowMajor> IntArray;
typedef Eigen::Array<float, 1, Eigen::Dynamic, Eigen::RowMajor> FloatArray;


#include "tree.cpp"
#include "merging.cpp"
#include "sampling.cpp"


PYBIND11_MODULE(_C, m) {
    m.doc() = "C++ implementation of bottom-up merging and top-down sampling for the paper \"Multi-granularity Interaction Simulation for Unsupervised Interactive Segmentation\".";
    m.def("bottom_up_merging", &bottom_up_merging, "");
    m.def("top_down_sampling", &top_down_sampling, "");
    m.def("get_leaf_descendents", &get_leaf_descendents, "");
    m.def("get_roots", &get_roots, "");
}
