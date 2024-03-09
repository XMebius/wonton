/**
  *******************************************************
  * @file           : TensorTest1.cpp
  * @author         : Mebius
  * @brief          : test for Tensor
  * @date           : 2024/3/9
  *******************************************************
  */

#include <Tensor.h>
#include <gtest/gtest.h>
#include <glog/logging.h>

TEST(TensorTest, construct1){
    wonton::Tensor<float> t(10);
    EXPECT_EQ(t.channels(), 1);
    EXPECT_EQ(t.rows(), 1);
    EXPECT_EQ(t.cols(), 10);
    EXPECT_EQ(t.size(), 10);
    std::vector<uint32_t> vec = {1,1,10};   // [channels, rows, cols]
    std::vector<uint32_t> raw_vec = {10};
    EXPECT_EQ(t.shapes(), vec);
    EXPECT_EQ(t.raw_shapes(), raw_vec);
    EXPECT_EQ(t.index(0), 0.0f);
    EXPECT_FALSE(t.empty());
}

TEST(TensorTest, construct2){
    wonton::Tensor<float> t(3, 4);
    EXPECT_EQ(t.channels(), 1);
    EXPECT_EQ(t.rows(), 3);
    EXPECT_EQ(t.cols(), 4);
    EXPECT_EQ(t.size(), 12);
    std::vector<uint32_t> vec = {1,3,4};   // [channels, rows, cols]
    std::vector<uint32_t> raw_vec = {3,4};
    EXPECT_EQ(t.shapes(), vec);
    EXPECT_EQ(t.raw_shapes(), raw_vec);
    EXPECT_EQ(t.index(11), 0.0f);
    EXPECT_FALSE(t.empty());
}

TEST(TensorTest, construct3){
    wonton::Tensor<float> t(3, 4, 5);
    EXPECT_EQ(t.channels(), 3);
    EXPECT_EQ(t.rows(), 4);
    EXPECT_EQ(t.cols(), 5);
    EXPECT_EQ(t.size(), 60);
    std::vector<uint32_t> vec = {3,4,5};   // [channels, rows, cols]
    std::vector<uint32_t> raw_vec = {3,4,5};
    EXPECT_EQ(t.shapes(), vec);
    EXPECT_EQ(t.raw_shapes(), raw_vec);
    EXPECT_EQ(t.index(59), 0.0f);
    EXPECT_FALSE(t.empty());
}

TEST(TensorTest, construct4){
    std::vector<float> vec = {3,4,5,3,4,5,3,4,5,3,4,5};   // [channels, rows, cols]
    wonton::Tensor<float> t(2, 2, 3);
    t.fill(vec, true);
    arma::fmat m = arma::fmat(1, 3);
    m = {3,4,5};
}

TEST(TensorTest, construct5){
    wonton::Tensor<float> t1(1,2,3);
    t1.fill(3);
    EXPECT_EQ(t1.index(0), 3);
    LOG(INFO) << "data is "<<t1.data();

    wonton::Tensor<float> t2(1,2,3);
    std::vector<float> vec = {3,2,1,1,2,3};
    t2.fill(vec, true);
    LOG(INFO) << "data is "<<t2.data();
    EXPECT_EQ(t1.shapes(), t2.shapes());
}

TEST(test_tensor, tensor_init1D) {
    using namespace wonton;
    Tensor<float> f1(4);
    f1.fill(1.f);
    const auto &raw_shapes = f1.raw_shapes();
    LOG(INFO) << "-----------------------Tensor1D-----------------------";
    LOG(INFO) << "raw shapes size: " << raw_shapes.size();
    const uint32_t size = raw_shapes.at(0);
    LOG(INFO) << "data numbers: " << size;
    f1.show();
}


TEST(test_tensor, tensor_init2D) {
    using namespace wonton;
    Tensor<float> f1(4, 4);
    f1.fill(1.f);

    const auto &raw_shapes = f1.raw_shapes();
    LOG(INFO) << "-----------------------Tensor2D-----------------------";
    LOG(INFO) << "raw shapes size: " << raw_shapes.size();
    const uint32_t rows = raw_shapes.at(0);
    const uint32_t cols = raw_shapes.at(1);

    LOG(INFO) << "data rows: " << rows;
    LOG(INFO) << "data cols: " << cols;
    f1.show();
}


TEST(test_tensor, tensor_init3D_3) {
    using namespace wonton;
    Tensor<float> f1(2, 3, 4);
    f1.fill(1.f);

    const auto &raw_shapes = f1.raw_shapes();
    LOG(INFO) << "-----------------------Tensor3D 3-----------------------";
    LOG(INFO) << "raw shapes size: " << raw_shapes.size();
    const uint32_t channels = raw_shapes.at(0);
    const uint32_t rows = raw_shapes.at(1);
    const uint32_t cols = raw_shapes.at(2);

    LOG(INFO) << "data channels: " << channels;
    LOG(INFO) << "data rows: " << rows;
    LOG(INFO) << "data cols: " << cols;
    f1.show();
}

TEST(test_tensor, tensor_init3D_2) {
    using namespace wonton;
    Tensor<float> f1(1, 2, 3);
    f1.fill(1.f);

    const auto &raw_shapes = f1.raw_shapes();
    LOG(INFO) << "-----------------------Tensor3D 2-----------------------";
    LOG(INFO) << "raw shapes size: " << raw_shapes.size();
    const uint32_t rows = raw_shapes.at(0);
    const uint32_t cols = raw_shapes.at(1);

    LOG(INFO) << "data rows: " << rows;
    LOG(INFO) << "data cols: " << cols;
    f1.show();
}

TEST(test_tensor, tensor_init3D_1) {
    using namespace wonton;
    Tensor<float> f1(1, 1, 3);
    f1.fill(1.f);

    const auto &raw_shapes = f1.raw_shapes();
    LOG(INFO) << "-----------------------Tensor3D 1-----------------------";
    LOG(INFO) << "raw shapes size: " << raw_shapes.size();
    const uint32_t size = raw_shapes.at(0);

    LOG(INFO) << "data numbers: " << size;
    f1.show();
}

TEST(test_fill_reshape, fill1) {
    using namespace wonton;
    Tensor<float> f1(2, 3, 4);
    std::vector<float> values(2 * 3 * 4);
    // 将1到12填充到values中
    for (int i = 0; i < 24; ++i) {
        values.at(i) = float(i + 1);
    }
    f1.fill(values);
    f1.show();
}
