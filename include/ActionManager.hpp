//
// Project: DooT2
// File: ActionManager.hpp
//
// Copyright (c) 2022 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once


#include "ActionConverter.hpp"
#include "MathTypes.hpp"

#include <random>
#include <functional>


class ActionManager {
public:
    struct UpdateParams {
        Vec2f               p                       {0.0f, 0.0f};   // player position
        Vec2f               v                       {0.0f, 0.0f};   // player velocity
        Vec2f               a                       {0.0f, 0.0f};   // player acceleration

        double              smoothing               {0.8f};
        double              sigma                   {2.0f};

        std::vector<float>  actionVectorOverwrite;
    };

    struct CallParams {
        Vec2f   playerPos   {0.0f, 0.0f};
    };


    ActionManager();

    template <typename T_Module>
    bool addModule(T_Module* module);

    template <typename T_Module>
    T_Module::State& getModuleState() const;

    void reset();

    gvizdoom::Action operator()(const CallParams& callParams);

    const std::vector<float>& getActionVector() const noexcept;

private:
    using ModuleCall = std::function<void(const CallParams&, UpdateParams&, ActionManager&)>;
    using ModuleReset = std::function<void()>;

    ActionConverter<float>                  _actionConverter;
    std::vector<float>                      _actionVector;
    std::vector<void*>                      _modules;
    std::vector<ModuleCall>                 _moduleCalls;
    std::vector<ModuleReset>                _moduleResets;
    UpdateParams                            _updateParams;

    Vec2f                                   _pPrev; // previous position
    Vec2f                                   _vPrev; // previous velocity

    static std::default_random_engine       _rnd;
    static std::normal_distribution<float>  _rndNormal;


    static int64_t moduleIdCounter;
    template <typename T_Module>
    static int64_t moduleId();

    void updateActionVector(const UpdateParams& params);
};


template<typename T_Module>
bool ActionManager::addModule(T_Module* module)
{
    auto mId = moduleId<T_Module>();
    _modules.resize(mId+1, nullptr);
    _moduleCalls.resize(mId+1);
    _moduleResets.resize(mId+1);

    assert((bool)_moduleCalls[mId] == (bool)_moduleResets[mId]); // indicates a bug
    if (_modules[mId] != nullptr) // existing module already in place, omit addition
        return false;

    _modules[mId] = module;
    _moduleCalls[mId] = std::bind(&T_Module::operator(), module,
        std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
    _moduleResets[mId] = std::bind(&T_Module::reset, module);

    return true;
}

template <typename T_Module>
T_Module::State& ActionManager::getModuleState() const
{
    auto mId = moduleId<T_Module>();

    // check that the requested module exists
    assert(_modules.size() > mId);
    assert(_modules[mId] != nullptr);

    // cast the stored module pointer to correct type and return its state
    return static_cast<T_Module*>(_modules[mId])->state;
}

template<typename T_Module>
int64_t ActionManager::moduleId()
{
    static int64_t id = moduleIdCounter++;
    return id;
}
