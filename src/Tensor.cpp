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
        if (channels == 1 && rows == 1) {
            this->raw_shape = std::vector<uint32_t>{cols};
        } else if (channels == 1) {
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
        return this->raw_data.size();
    }

    std::vector<uint32_t> Tensor<float>::shapes() const {
        return {this->channels(), this->rows(), this->cols()};  // [n_slices, n_rows, n_cols]
    }

    const std::vector<uint32_t> &Tensor<float>::raw_shapes() const {
        CHECK(!this->raw_shape.empty());
        CHECK_LE(this->raw_shape.size(), 3);
        CHECK_GE(this->raw_shape.size(), 1);
        return this->raw_shape;
    }

    float Tensor<float>::index(uint32_t offset) const {
        CHECK(!this->raw_data.empty());
        CHECK_LT(offset, this->size()) << "offset is out of range";
        return this->raw_data.at(offset);
    }

    float &Tensor<float>::index(uint32_t offset) {
        CHECK(!this->raw_data.empty());
        CHECK_LT(offset, this->size()) << "offset is out of range";
        return this->raw_data.at(offset);
    }

    bool Tensor<float>::empty() const {
        return this->raw_data.empty();
    }

    void Tensor<float>::set_data(const arma::fcube &data) {
        CHECK(data.n_rows == this->raw_data.n_rows) << "rows is not equal";
        CHECK(data.n_cols == this->raw_data.n_cols) << "cols is not equal";
        CHECK(data.n_slices == this->raw_data.n_slices) << "channels is not equal";
        this->raw_data = data;
    }

    arma::fcube &Tensor<float>::data() {
        return this->raw_data;
    }

    const arma::fcube &Tensor<float>::data() const {
        return this->raw_data;
    }

    arma::fmat &Tensor<float>::slice(uint32_t channel) {
        CHECK_LE(channel, this->channels()) << "channel is out of range";
        return this->raw_data.slice(channel);
    }

    const arma::fmat &Tensor<float>::slice(uint32_t channel) const {
        CHECK_LE(channel, this->channels()) << "channel is out of range";
        return this->raw_data.slice(channel);
    }

    float Tensor<float>::at(uint32_t channel, uint32_t row, uint32_t col) const {
        CHECK_LE(channel, this->channels()) << "channel is out of range";
        CHECK_LE(row, this->rows()) << "row is out of range";
        CHECK_LE(col, this->cols()) << "col is out of range";
        return this->raw_data.at(row, col, channel);
    }

    float &Tensor<float>::at(uint32_t channel, uint32_t row, uint32_t col) {
        CHECK_LE(channel, this->channels()) << "channel is out of range";
        CHECK_LE(row, this->rows()) << "row is out of range";
        CHECK_LE(col, this->cols()) << "col is out of range";
        return this->raw_data.at(row, col, channel);
    }

    void Tensor<float>::fill(float value) {
        CHECK(!this->raw_data.empty());
        this->raw_data.fill(value);
    }

    void Tensor<float>::fill(std::vector<float> values, bool row_major) {
        CHECK(!this->raw_data.empty());
        CHECK_EQ(values.size(), this->size()) << "values size is not equal to tensor size";
        if (row_major) {
            const uint32_t rows = this->rows();
            const uint32_t cols = this->cols();
            const uint32_t channels = this->channels();
            const uint32_t planes = rows * cols;

            for (uint32_t i = 0; i < channels; i++) {
                auto &channel_data = this->raw_data.slice(i);
                const arma::fmat &channel_data_t = arma::fmat(
                        values.data() + i * planes,
                        cols,
                        rows);
                channel_data = channel_data_t.t();
            }
        } else {
            std::copy(values.begin(), values.end(), this->raw_data.memptr());
        }
    }

    void Tensor<float>::show() {
        for (uint32_t i = 0; i < this->channels(); ++i) {
            LOG(INFO) << "Channel: " << i;
            LOG(INFO) << "\n" << this->raw_data.slice(i);
        }
    }

    std::vector<float> Tensor<float>::values(bool row_major) {
        CHECK(!this->raw_data.empty());
        std::vector<float> values(this->raw_data.size());
        if (!row_major) {
            std::copy(this->raw_data.begin(), this->raw_data.end(), values.begin());
        } else {
            uint32_t index = 0;
            for (uint32_t channel = 0; channel < this->raw_data.n_slices; channel++) {
                const arma::fmat &channel_data = this->raw_data.slice(channel);
                std::copy(channel_data.begin(), channel_data.end(), values.begin() + index);
                index += channel_data.size();
            }
            CHECK_EQ(index, values.size());
        }
        return values;
    }

    void Tensor<float>::ones() {
        CHECK(!this->raw_data.empty());
        this->fill(1.0f);
    }

    void Tensor<float>::zeros() {
        CHECK(!this->raw_data.empty());
        this->fill(0.0f);
    }

    void Tensor<float>::rand() {
        CHECK(!this->raw_data.empty());
        this->raw_data.randn();
    }

    void Tensor<float>::reshape(const std::vector<uint32_t> &shapes, bool row_major) {
        CHECK(!shapes.empty() && shapes.size() <= 3 && !this->raw_data.empty());
        const uint32_t origin_size = this->size();
        const uint32_t current_size = std::accumulate(
                shapes.begin(),
                shapes.end(),
                1,
                std::multiplies());
        CHECK(shapes.size() <= 3);
        CHECK(current_size == origin_size);

        std::vector<float> values;
        if (row_major) {
            values = this->values(true);
        }
        if (shapes.size() == 3) {
            this->raw_data.reshape(shapes.at(1), shapes.at(2), shapes.at(0));
            this->raw_shape = {shapes.at(0), shapes.at(1), shapes.at(2)};
        } else if (shapes.size() == 2) {
            this->raw_data.reshape(shapes.at(0), shapes.at(1), 1);
            this->raw_shape = {shapes.at(0), shapes.at(1)};
        } else {
            this->raw_data.reshape(1, shapes.at(0), 1);
            this->raw_shape = {shapes.at(0)};
        }

        if (row_major) {
            this->fill(values, true);
        }
    }

    void Tensor<float>::transform(const std::function<float(float)> &filter) {
        CHECK(!this->raw_data.empty());
        this->raw_data.transform(filter);
    }

    void Tensor<float>::flatten(bool row_major) {
        CHECK(!this->raw_data.empty());
        const uint32_t size = this->size();
        this->reshape({size}, row_major);
    }

    void Tensor<float>::padding(const std::vector<uint32_t> &pads, float padding_value) {
        CHECK(!this->raw_data.empty());
        CHECK_EQ(pads.size(), 4) << "pads size is not equal to 4";
        uint32_t pad_rows1 = pads.at(0);  // up
        uint32_t pad_rows2 = pads.at(1);  // bottom
        uint32_t pad_cols1 = pads.at(2);  // left
        uint32_t pad_cols2 = pads.at(3);  // right

        arma::Cube<float> new_data(this->raw_data.n_rows + pad_rows1 + pad_rows2,
                                   this->raw_data.n_cols + pad_cols1 + pad_cols2,
                                   this->raw_data.n_slices);
        new_data.fill(padding_value);

        new_data.subcube(pad_rows1, pad_cols1, 0, new_data.n_rows - pad_rows2 - 1,
                         new_data.n_cols - pad_cols2 - 1, new_data.n_slices - 1) = this->raw_data;
        this->raw_data = std::move(new_data);
        this->raw_shape = std::vector<uint32_t>{this->channels(), this->rows(), this->cols()};
    }
}
