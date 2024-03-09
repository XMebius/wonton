#include <iostream>
#include <gtest/gtest.h>
#include <glog/logging.h>

int main(int argc, char **argv) {
    std::cout << "Running main() from main.cpp" << std::endl;
    testing::InitGoogleTest(&argc, argv);
    google::InitGoogleLogging("wonton_1");
    FLAGS_alsologtostderr = true;

    LOG(INFO) << "START TEST ...\n";
    return RUN_ALL_TESTS();
}
