/**
  *******************************************************
  * @file           : TensorTest2.cpp
  * @author         : Mebius
  * @brief          : None
  * @date           : 2024/3/10
  *******************************************************
  */
#include <Test.h>

TEST(test_fill_reshape, reshape1) {
    using namespace wonton;
    LOG(INFO) << "-------------------Reshape-------------------";
    Tensor<float> f1(2, 3, 4);
    std::vector<float> values(2 * 3 * 4);
    // 将1到12填充到values中
    for (int i = 0; i < 24; ++i) {
        values.at(i) = float(i + 1);
    }
    f1.fill(values);
    f1.show();
    /// 将大小调整为(4, 3, 2)
    f1.reshape({4, 3, 2}, true);
    LOG(INFO) << "-------------------After Reshape-------------------";
    f1.show();
}

float MinusOne(float value) { return value - 1.f; }
TEST(test_transform, transform1) {
    using namespace wonton;
    Tensor<float> f1(2, 3, 4);
    f1.rand();
    LOG(INFO) << "-------------------before transforming-------------------";
    f1.show();
    f1.transform(MinusOne);
    LOG(INFO) << "-------------------after transforming-------------------";
    f1.show();
}

TEST(test_homework, homework1_flatten1) {
    using namespace wonton;
    Tensor<float> f1(2, 3, 4);
    f1.flatten(true);
    ASSERT_EQ(f1.raw_shapes().size(), 1);
    ASSERT_EQ(f1.raw_shapes().at(0), 24);
}

TEST(test_homework, homework1_flatten2) {
    using namespace wonton;
    Tensor<float> f1(12, 24);
    f1.flatten(true);
    ASSERT_EQ(f1.raw_shapes().size(), 1);
    ASSERT_EQ(f1.raw_shapes().at(0), 24 * 12);
}

TEST(test_homework, homework2_padding1) {
    using namespace wonton;
    Tensor<float> tensor(3, 4, 5);
    ASSERT_EQ(tensor.channels(), 3);
    ASSERT_EQ(tensor.rows(), 4);
    ASSERT_EQ(tensor.cols(), 5);

    tensor.fill(1.f);
    tensor.padding({1, 2, 3, 4}, 0);
    ASSERT_EQ(tensor.rows(), 7);
    ASSERT_EQ(tensor.cols(), 12);

    int index = 0;
    for (int c = 0; c < tensor.channels(); ++c) {
        for (int r = 0; r < tensor.rows(); ++r) {
            for (int c_ = 0; c_ < tensor.cols(); ++c_) {
                if ((r >= 2 && r <= 4) && (c_ >= 3 && c_ <= 7)) {
                    ASSERT_EQ(tensor.at(c, r, c_), 1.f) << c << " "
                                                        << " " << r << " " << c_;
                }
                index += 1;
            }
        }
    }
}

TEST(test_homework, homework2_padding2) {
    using namespace wonton;
    ftensor tensor(3, 4, 5);
    ASSERT_EQ(tensor.channels(), 3);
    ASSERT_EQ(tensor.rows(), 4);
    ASSERT_EQ(tensor.cols(), 5);

    tensor.fill(1.f);
    tensor.padding({2, 2, 2, 2}, 3.14f);
    ASSERT_EQ(tensor.rows(), 8);
    ASSERT_EQ(tensor.cols(), 9);

    int index = 0;
    for (int c = 0; c < tensor.channels(); ++c) {
        for (int r = 0; r < tensor.rows(); ++r) {
            for (int c_ = 0; c_ < tensor.cols(); ++c_) {
                if (c_ <= 1 || r <= 1) {
                    ASSERT_EQ(tensor.at(c, r, c_), 3.14f);
                } else if (c >= tensor.cols() - 1 || r >= tensor.rows() - 1) {
                    ASSERT_EQ(tensor.at(c, r, c_), 3.14f);
                }
                if ((r >= 2 && r <= 5) && (c_ >= 2 && c_ <= 6)) {
                    ASSERT_EQ(tensor.at(c, r, c_), 1.f);
                }
                index += 1;
            }
        }
    }
}