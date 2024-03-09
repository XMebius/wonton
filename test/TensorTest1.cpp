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
}