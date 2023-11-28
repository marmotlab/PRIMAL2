# include <pybind11/pybind11.h>
# include <pybind11/stl.h>

# include "new_driver.cpp"

namespace py = pybind11;


PYBIND11_MODULE(cbs_py, m) {

    m.doc() = "example plugin";
    m.def("findPath_new", &findPath_new, "add two integers");
}