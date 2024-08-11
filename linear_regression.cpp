#include <vector>
#include <iostream>
#include <ostream>
#include <type_traits>

namespace TypeChecks
{
    using std::begin;
    using std::end;

    template <typename T>
    concept is_iterable = requires(T &t) {
        std::begin(t) != std::end(t);
        ++std::begin(t);
        *std::begin(t);
    };

    template <typename I>
    concept is_numeric = requires(I &i) {
        { i + 0 } -> std::same_as<I>;
        { i - 0 } -> std::same_as<I>;
        { i * 1 } -> std::same_as<I>;
        { i / 1 } -> std::same_as<I>;
        { ++i   } -> std::same_as<I &>;
        { i++   } -> std::same_as<I>; 
    };
}

namespace ML
{
    typedef std::vector<float> floatVector;

    enum Ops {
        ADD,
        SUB,
        MULTIPLY,
        DIVIDE
    };

    // Y = M * X + C
    class LinearRegression {

    public:

        floatVector inputs;
        floatVector outputs;
        float slope;
        float intercept;
        float learningRate;

        LinearRegression(
            floatVector i,
            floatVector o,
            float m = 1,
            float c = 1,
            float lr = 0.001
        ) : inputs(i), outputs(o), slope(m), intercept(c), learningRate(lr) {}


        float operate(float v1, float v2, Ops op){
            switch (op) 
            {
                case Ops::ADD:
                    return v1 + v2;
                case Ops::SUB:
                    return v1 - v2;
                case Ops::MULTIPLY:
                    return v1 * v2;
                case Ops::DIVIDE:
                    if (v2 == 0.0f) throw "Division by zero !";
                    return v1 / v2;
                default:
                    throw "Unkonwn operation passed !";
            }
        }

        template <typename T> // val can be a number or a vector
        floatVector operateOnVectorValuesWith(const T &val, const floatVector &mainVector, Ops operation) 
        {
            floatVector res;
            // Scaler multiplication
            if constexpr (std::is_arithmetic<T>::value) {    
                for (auto ele : mainVector) {
                    result.push_back(operate(ele, val, operation));
                }
            }
            // Vector multiplication
            else 
            { 
                static_assert(
                    std::is_same<T, std::vector<float>>::value, 
                    "val must be either a number or a vector of floats"
                );

                if (val.size() == mainVector.size()) {
                    for (size_t i = 0; i < mainVector.size(); ++i) {
                        result.push_back(operate(mainVector[i], val[i], operation)); //mainVector[i] * val[i]
                    }
                } 
                else {
                    throw std::invalid_argument("Vector lengths don't match!");
                }
            }

            return res;
        }

        float mean(floatVector vec){
            float sum = 0.0;
            for(auto v: vec){
                sum += v;
            }
            return sum / vec.size();
        }

        floatVector predict(float sl=0.0, float in=0.0){
            auto slope = (sl == 0.0) ? this->slope : sl;
            auto intercept = (in == 0.0) ? this->intercept : in;
            auto temp = operateOnVectorValuesWith(intercept, inputs, Ops::MULTIPLY);
            return operateOnVectorValuesWith(slope, temp, Ops::ADD);
        }

        float getMSELoss(floatVector predictions){
            auto temp = operateOnVectorValuesWith(outputs, predictions, Ops::SUB);
            auto tempSq = operateOnVectorValuesWith(temp, temp, Ops::MULTIPLY);
            return mean(tempSq);
        }

        void backwardPropagationBGD(floatVector predictions){
            auto diff = operateOnVectorValuesWith(predictions, outputs, Ops::SUB);
            auto temp = operateOnVectorValuesWith(inputs, diff, Ops::MULTIPLY);
            float gradientSlope = 2 * mean(temp);
            float gradientIntercept = 2 * mean(diff);
            slope -= learningRate * gradientSlope;
            intercept -= learningRate * gradientIntercept;
        }

        void train(int epochs){
            for(int i = 0; i < epochs; i++){
                auto predictions = predict();
                backwardPropagationBGD(predictions);
                std::cout << "EPOCH: " << i << ", MSE Loss: " << getMSELoss(predictions) << std::endl;
            }
        }
        
    };
};