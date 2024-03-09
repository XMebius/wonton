/**
  *******************************************************
  * @file           : Tensor.cpp
  * @author         : Mebius
  * @brief          : None
  * @date           : 2024/3/9
  *******************************************************
  */

#include <Tensor.h>
#include <glog/logging.h>

namespace wonton {
    Tensor<float>::Tensor(uint32_t length) {
        this->raw_data = arma::fcube(1, length, 1); // [n_rows, n_cols, n_slices]
        this->raw_shape = std::vector<uint32_t>{length};
    }

    Tensor<float>::Tensor(uint32_t rows, uint32_t cols) {
        this->raw_data = arma::fcube(rows, cols, 1);
        if (rows == 1) {
            this->raw_shape = std::vector<uint32_t>{cols};
        } else {
            this->raw_shape = std::vector<uint32_t>{rows, cols};
        }
    }

    Tensor<float>::Tensor(uint32_t channels, uint32_t rows, uint32_t cols) {
        this->raw_data = arma::fcube(rows, cols, channels);
        if (channels == 1 && rows == 1) {
            this->raw_shape = std::vector<uint32_t>{cols};
        } else if (channels == 1) {
            this->raw_shape = std::vector<uint32_t>{rows, cols};
        } else {
            this->raw_shape = std::vector<uint32_t>{channels, rows, cols};
        }
    }

    Tensor<float>::Tensor(std::vector<uint32_t> shapes) {
        CHECK(!shapes.empty() && shapes.size() <= 3);
        uint32_t remaining = 3 - shapes.size();
        std::vector<uint32_t> shapes_(3, 1);  // [1, 1, 1]
        std::copy(shapes.begin(), shapes.end(), shapes_.begin() + remaining);

        uint32_t channels = shapes_[0];
        uint32_t rows = shapes_[1];
        uint32_t cols = shapes_[2];

        this->raw_data = arma::fcube(rows, cols, channels);
        if(channels == 1&&rows==1){
            this->raw_shape = std::vector<uint32_t>{cols};
        } else if(channels == 1){
            this->raw_shape = std::vector<uint32_t>{rows, cols};
        } else {
            this->raw_shape = std::vector<uint32_t>{channels, rows, cols};
        }
    }

    uint32_t Tensor<float>::rows() const {
        CHECK(!this->raw_data.empty());
        return this->raw_data.n_rows;
    }

    uint32_t Tensor<float>::cols() const {
        CHECK(!this->raw_data.empty());
        return this->raw_data.n_cols;
    }

    uint32_t Tensor<float>::channels() const {
        CHECK(!this->raw_data.empty());
        return this->raw_data.n_slices;
    }

    uint32_t Tensor<float>::size() const {
        CHECK(!this->raw_data.empty());
        return this->raw_data.n_elem;
    }

    std::vector<uint32_t> Tensor<float>::shapes() const {
        return {this->channels(), this->rows(), this->cols()};  // [n_slices, n_rows, n_cols]
    }

    const std::vector<uint32_t>& Tensor<float>::raw_shapes() const {
        CHECK(!this->raw_shape.empty());
        CHECK_LE(this->raw_shape.size(), 3);
        CHECK_GE(this->raw_shape.size(), 1);
        return this->raw_shape;
    }
}
