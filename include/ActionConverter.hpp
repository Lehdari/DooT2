//
// Project: DooT2
// File: ActionConverter.hpp
//
// Copyright (c) 2022 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once


#include <unordered_map>
#include <vector>
#include <algorithm>

#include "gvizdoom/Action.hpp"


template <typename T_Scalar>
class ActionConverter {
public:
    ActionConverter(T_Scalar min = -1.0, T_Scalar max = 1.0, int minAngle = -1000, int maxAngle = 1000) noexcept;

    // Bind action key to action vector index
    void setKeyIndex(size_t id, const gvizdoom::Action::Key& key);

    // Bind angle to action vector index
    void setAngleIndex(size_t id);

    // Convert a scalar-valued action vector to a gvizdoom Action
    gvizdoom::Action operator()(const std::vector<T_Scalar>& actionVector) const noexcept;

    // Convert a gvizdoom Action to a scalar-valued action vector
    std::vector<T_Scalar> operator()(const gvizdoom::Action& action,
        size_t actionVectorLength) const noexcept;

private:
    T_Scalar    _min;
    T_Scalar    _max;
    int         _minAngle;
    int         _maxAngle;

    std::unordered_map<size_t, gvizdoom::Action::Key>   _keyMap;
    size_t                                              _angleMap;
};


template<typename T_Scalar>
ActionConverter<T_Scalar>::ActionConverter(T_Scalar min, T_Scalar max, int minAngle, int maxAngle) noexcept :
    _min        (min),
    _max        (max),
    _minAngle   (minAngle),
    _maxAngle   (maxAngle)
{
}

template<typename T_Scalar>
void ActionConverter<T_Scalar>::setKeyIndex(size_t id, const gvizdoom::Action::Key& key)
{
    _keyMap[id] = key;
}

template<typename T_Scalar>
void ActionConverter<T_Scalar>::setAngleIndex(size_t id)
{
    _keyMap.erase(id); // unbind key if one is set
    _angleMap = id;
}

template<typename T_Scalar>
gvizdoom::Action ActionConverter<T_Scalar>::operator()(const std::vector<T_Scalar>& actionVector) const noexcept
{
    T_Scalar middle = (_min+_max) / 2;

    gvizdoom::Action action;
    for (size_t i=0; i<actionVector.size(); ++i) {
        auto& v = actionVector[i];
        if (_keyMap.contains(i) && v >= middle) {
            action.set(_keyMap.at(i));
        }
        else if (i == _angleMap) {
            float angle = std::clamp<T_Scalar>(v, _min, _max);
            action.setAngle(((angle-_min)/(_max-_min))*(_maxAngle-_minAngle)+_minAngle);
        }
    }

    return action;
}

template<typename T_Scalar>
std::vector<T_Scalar> ActionConverter<T_Scalar>::operator()(const gvizdoom::Action& action,
    size_t actionVectorLength) const noexcept
{
    std::vector<T_Scalar> actionVector(actionVectorLength, _min);

    for (size_t i = 0; i < actionVector.size(); ++i) {
        if (_keyMap.contains(i) && action.isSet(_keyMap.at(i))) {
            actionVector[i] = _max;
        } else if (_angleMap == i) {

            auto angle = static_cast<T_Scalar>(action.angle());
            actionVector[i] = (angle - _minAngle) / (_maxAngle-_minAngle) * (_max - _min) + _min;
        }
    }

    return actionVector;
}
