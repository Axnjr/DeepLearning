#include <vector>
#include <type_traits>

namespace SNN {

    typedef std::vector<float> floatVector;
    template <typename T> typename std::enable_if<std::is_arithmetic<T>::value, T>::type numbers;

    class LinearRegression {
        // Y = M * X + C
        public:

            floatVector inputs;
            floatVector outputs;
            int slope;
            int constant;

            LinearRegression(
                floatVector i, 
                floatVector o,
                int m = 1,
                int c = 1
            ) : inputs(i), outputs(o), slope(m), constant(c) {}

        floatVector multiplyVectorValuesWith(floatVector v){

        }

        floatVector forwardPropagation(){

        }

    };

};