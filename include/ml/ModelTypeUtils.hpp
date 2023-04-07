//
// Project: DooT2
// File: ModelTypeUtils.hpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once

#include <type_traits>
#include <string>
#include <stdexcept>

#include "util/TypeCounter.hpp"


// List all available model types in this macro
// Remember to add the include in Models.hpp too!
#define ML_MODEL_TYPES               \
    ML_MODEL_TYPE(AutoEncoderModel)  \
    ML_MODEL_TYPE(EncoderModel)      \
    ML_MODEL_TYPE(RandomWalkerModel)


namespace ml {

// Forward declarations
#define ML_MODEL_TYPE(MODEL) class MODEL;
ML_MODEL_TYPES
#undef ML_MODEL_TYPE

}

namespace detail {

// Generate the type counter for IDs
#define ML_MODEL_TYPE(MODEL) ml::MODEL,
using ModelTypeCounter = TypeCounter<ML_MODEL_TYPES void>;
#undef ML_MODEL_TYPE

} // namespace detail

namespace ml {

// Type info structs (mapping from model type to parameters)
template <typename T_Model>
struct ModelTypeInfo {};

#define ML_MODEL_TYPE(MODEL)                                                 \
template <>                                                                  \
struct ModelTypeInfo<MODEL> {                                                \
    static constexpr int    id      {detail::ModelTypeCounter::Id<MODEL>()}; \
    static constexpr char   name[]  {#MODEL};                                \
};
ML_MODEL_TYPES
#undef ML_MODEL_TYPE

inline std::string modelTypeName(int modelTypeId)
{
    switch (modelTypeId) {
        #define ML_MODEL_TYPE(MODEL)           \
        case ModelTypeInfo<MODEL>::id:         \
            return ModelTypeInfo<MODEL>::name;
        ML_MODEL_TYPES
        #undef ML_MODEL_TYPE
        default: break;
    }
    throw std::runtime_error("Invalid model type id: " + std::to_string(modelTypeId) +
        " - have all model types been listed in ModelTypeUtils.hpp?");
}

// Callbacks for fetching model type using type name or ID,
// intended to be used with a generic (templated) lambda:
// modelTypeCallback(modelTypeName, []<typename T>(){ /* T is the model type here */ });
template <typename F>
void modelTypeNameCallback(const std::string& typeName, F&& f)
{
    #define ML_MODEL_TYPE(MODEL)                \
    if (typeName == ModelTypeInfo<MODEL>::name) \
        return f.template operator()<MODEL>();
    ML_MODEL_TYPES
    #undef ML_MODEL_TYPE
    throw std::runtime_error("No type name match for \"" + typeName + "\" found");
}

template <typename F>
void modelTypeIdCallback(int typeId, F&& f)
{
    #define ML_MODEL_TYPE(MODEL)              \
    if (typeId == ModelTypeInfo<MODEL>::id)   \
        return f.template operator()<MODEL>();
    ML_MODEL_TYPES
    #undef ML_MODEL_TYPE
    throw std::runtime_error("No type name match for " + std::to_string(typeId) + " found");
}

// similar to the functions above, but this function calls the callback for each model type
template <typename F>
void modelForEachTypeCallback(F&& f)
{
    #define ML_MODEL_TYPE(MODEL)    \
    f.template operator()<MODEL>();
    ML_MODEL_TYPES
    #undef ML_MODEL_TYPE
}

} // namespace ml

#undef ML_MODEL_TYPES
