# 开发笔记
[toc]

## 1. explicit关键字
概述：explicit关键字只能用于类内部的构造函数，用于修饰只有一个参数的构造函数，表示该构造函数是显示的，禁止隐式转换。

构造函数有三种情况：
1. 默认构造函数
2. 有参构造函数
3. 拷贝构造函数
三种方式举例如下
```cpp
class A
{
public:
    A(){} // 默认构造函数
    A(int a){} // 有参构造函数
    A(const A& a){} // 拷贝构造函数
};
```
所谓拷贝构造的不同之处在于，拷贝构造函数的参数是一个const引用，而不是一个对象。
而所谓的隐式转换是指，编译器会自动调用构造函数，将一个类型转换为另一个类型。例如：
```cpp
class A
{
public:
    A(int a){}; // 有参构造函数
};
void func(A a){}
int main()
{
    A a = 10; // 隐式转换1
    func(10); // 隐式转换2
}
```
为了控制隐式转换的发生，可以将构造函数声明为explicit，如下：
```cpp
class A
{
public:
    explicit A(int a){}; // 有参构造函数
};
void func(A a){}
int main()
{
    A a = 10; // 错误
    func(10); // 错误
    A a(10); // 正确
}
```

## 2. 返回引用值

下面两个代码的区别是什么：
```cpp
std::vector<uint32_t> Tensor<float>::shapes() const {
    return {this->channels(), this->rows(), this->cols()};  // [n_slices, n_rows, n_cols]
}

const std::vector<uint32_t>& Tensor<float>::raw_shapes() const {
    CHECK(!this->raw_shape.empty());
    CHECK_LE(this->raw_shape.size(), 3);
    CHECK_GE(this->raw_shape.size(), 1);
    return this->raw_shape;
}
```
正常来说，返回引用效率更高，可以避免复制整个vector。同时const保证了后续在调用raw_shapes()的时候不会修改raw_shape。将张量的shape设置为const是合理的因为这个数值应该避免被修改。

而第一个函数不能设置引用，因为`{this->channels(), this->rows(), this->cols()}`是一个临时变量，返回引用会导致返回的引用指向一个临时变量，这个临时变量在函数结束后会被销毁，引用指向的内存地址会被释放，这样返回的引用就是一个野指针了。
