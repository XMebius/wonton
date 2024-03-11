/**
  *******************************************************
  * @file           : Tensor.h
  * @author         : Mebius
  * @brief          : declare namespace and class for tensor
  * @date           : 2024/3/9
  *******************************************************
  */


#ifndef WONTON_TENSOR_H
#define WONTON_TENSOR_H

#include <armadillo>
#include <vector>

namespace wonton{
    template<typename T> class Tensor {};

    template<> class Tensor<double> {};
    template<> class Tensor<float> {
    public:
        /// constructors
        Tensor() = default; // default constructor
        Tensor(const Tensor& ) = default; // copy constructor
        Tensor(Tensor&& ) = default; // move constructor
        /**
         * @brief Construct a Tensor of 1 dim
         * @param length
         */
        Tensor(uint32_t length);
        /**
         * @brief Construct a Tensor of 2 dim
         * @param rows
         * @param cols
         */
        Tensor(uint32_t rows, uint32_t cols);
        /**
         * @brief Construct a Tensor of 3 dim
         * @param rows
         * @param cols
         * @param slices
         */
        Tensor(uint32_t channels, uint32_t rows, uint32_t cols);
        /**
         * @brief construct a Tensor of n<=3 dim
         * @param shape: shape of the tensor(<=3)
         */
        Tensor(std::vector<uint32_t> shape);

        /// member operator
        Tensor& operator=(Tensor&& ) = default; // move assignment
        Tensor& operator=(const Tensor& ) = default; // copy assignment

        /// destructor
        ~Tensor() = default;

        /// member function
        /**
         * @brief return the row of the tensor
         * @return
         */
        uint32_t rows() const;
        /**
         * @brief return the col of the tensor
         * @return
         */
        uint32_t cols() const;
        /**
         * @brief return the channels of the tensor
         * @return
         */
        uint32_t channels() const;
        /**
         * @brief return the element numbers of the tensor
         * @return
         */
        uint32_t size() const;
        /**
         * @brief return the shape of the tensor
         * @return
         */
        std::vector<uint32_t>shapes() const;
        /**
         * @brief return the original shape of the tensor
         * @return
         */
        const std::vector<uint32_t>& raw_shapes() const;
        /**
         * @brief get data in offset position
         * @param offset
         * @return
         */
        float index(uint32_t offset) const;
        float& index(uint32_t offset);
        /**
         * @brief check empty
         * @return
         */
        bool empty() const;
        /**
         * @brief set data from a 3-dim matrix
         * @param data
         */
        void set_data(const arma::fcube& data);
        /**
         * @brief get data values
         * @return
         */
        arma::fcube& data();
        const arma::fcube& data() const;
        /**
         * @brief get data values from a channel
         * @param channel
         * @return
         */
        arma::fmat& slice(uint32_t channel);
        const arma::fmat& slice(uint32_t channel) const;
        /**
         * @brief get data values from a channel, row and col
         * @param channel
         * @param row
         * @param col
         * @return
         */
        float at(uint32_t channel, uint32_t row, uint32_t col) const;
        float& at(uint32_t channel, uint32_t row, uint32_t col);
        /**
         * @brief fill the tensor with a value
         * @param value
         */
        void fill(float value);
        void fill(std::vector<float> values, bool row_major = true);

        // TODO
        /**
         * @brief get data values through row_major or not
         * @param row_major
         * @return
         */
        std::vector<float> values(bool row_major);
        /**
         * @brief show the tensor
         */
        void show();
        /**
         * @brief set all the elements to 1
         */
        void ones();
        /**
         * @brief et all the elements to 0
         */
        void zeros();
        /**
         * @brief set all the elements to a random value
         */
        void rand();
        /**
         * @brief reshape the tensor
         * @param shape : new shape
         * @param row_major
         */
        void reshape(const std::vector<uint32_t>& shape, bool row_major);
        /**
         * @brief filter the elements through a function
         * @param filter
         */
        void transform(const std::function<float(float)>& filter);
        /**
         * @brief flatten the tensor
         * @param row_major
         */
        void flatten(bool row_major);
        /**
         * @brief padding the tensor
         * @param pads : padding size
         * @param padding_value : padding value
         */
        void padding(const std::vector<uint32_t>& pads,float padding_value);



    private:
        std::vector<uint32_t> raw_shape;     // original shape
        arma::fcube raw_data;                // original data (always 3-dim)
    };
    template<> class Tensor<uint8_t> {};    // 8-bit unsigned integer
    using ftensor = Tensor<float>;
};

#endif //WONTON_TENSOR_H
