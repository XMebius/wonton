#include <armadillo>
#include <gtest/gtest.h>
#include <glog/logging.h>

TEST(ArmaTest, add) {
    arma::vec a = {1, 2, 3};
    arma::vec b = {4, 5, 6};
    arma::vec c = a + b;
    EXPECT_EQ(c(0), 5);
    EXPECT_EQ(c(1), 7);
    EXPECT_EQ(c(2), 9);
}

TEST(ArmTest, sub) {
    arma::vec a = {1, 2, 3};
    arma::vec b = {4, 5, 6};
    arma::vec c = a - b;
    EXPECT_EQ(c(0), -3);
    EXPECT_EQ(c(1), -3);
    EXPECT_EQ(c(2), -3);
}

TEST(ArmTest, mul) {
    arma::vec a = {1, 2, 3};
    arma::vec b = {4, 5, 6};
    arma::vec c = a % b;
    EXPECT_EQ(c(0), 4);
    EXPECT_EQ(c(1), 10);
    EXPECT_EQ(c(2), 18);
}

TEST(ArmTest, div) {
    arma::vec a = {1, 2, 3};
    arma::vec b = {4, 5, 6};
    arma::vec c = a / b;
    EXPECT_EQ(c(0), 0.25);
    EXPECT_EQ(c(1), 0.4);
    EXPECT_EQ(c(2), 0.5);
}