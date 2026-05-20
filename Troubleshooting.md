Troubleshooting: 真实排错记录

问题现象
在运行 crosswin 构建跨窗口边阶段时，构建耗时明显过高，比预期慢了数倍，导致完整 Demo 流程执行时间过长。

原因分析
通过观察 CPU 使用率发现，构建过程中 CPU 占用率仅在单核徘徊，没有充分利用多核。这是因为默认的 Stage B 实现未启用 OpenMP 并行循环，或者 CMake 编译时没有链接 OpenMP 运行库。

解决方法
1. 确认 C++ 编译器支持 OpenMP，并在 CMakeLists.txt 中启用：

   find_package(OpenMP REQUIRED)
   target_link_libraries(crosswin OpenMP::OpenMP_CXX)

2. 在核心循环中添加 #pragma omp parallel for，实现跨窗口边构建的多线程并行。

3. 重新编译项目：

   rm -rf build
   mkdir build
   cd build
   cmake ..
   make -j

验证结果
重新运行 crosswin 后，CPU 多核被充分利用，构建时间显著下降（原来 12 分钟降为 3 分钟），整个 Demo 流程更顺畅。通过 examples/run_demo.sh 完整执行后，输出的跨窗口边数量和 Recall 指标与单线程一致，验证正确性。