//
// Project: DooT2
// File: ExperimentName.cpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "util/ExperimentUtils.hpp"
#include "Constants.hpp"

#include <sstream>
#include <regex>


std::string formatExperimentName(const std::string& name, const nlohmann::json& modelConfig)
{
    std::string experimentName(name);

    // Replace {time} with GMT timestamp
    std::string timestamp = [](){
        using namespace std::chrono;
        std::stringstream ss;
        auto now = system_clock::to_time_t(system_clock::now());
        ss << std::put_time(std::gmtime(&now), "%Y%m%dT%H%M%S");
        return ss.str();
    }();
    experimentName = std::regex_replace(experimentName, std::regex("\\{time}"), timestamp);

    // Replace {version} with git version hash
    experimentName = std::regex_replace(experimentName, std::regex("\\{version}"), GIT_VERSION);

    // Replace all the parameter macros ({p:<parameter name>}) with the respective parameter values
    std::regex paramRegex("\\{p:(.*?)}");
    std::stringstream newExperimentNameSs;
    std::string::const_iterator it = experimentName.cbegin(), end = experimentName.cend();
    for (std::smatch match; std::regex_search(it, end, match, paramRegex); it = match[0].second)
    {
        newExperimentNameSs << match.prefix();
        std::string param = match[1].str();
        if (!modelConfig.contains(param)) {
            printf("WARNING: Model config does not contain parameter \"%s\", ignoring the macro\n",
                param.c_str()); // TODO logging
            newExperimentNameSs << match.str(); // replace here
        }
        else {
            if (modelConfig[param].type() == nlohmann::json::value_t::string)
                newExperimentNameSs << modelConfig[param].get<std::string>();
            else
                newExperimentNameSs << modelConfig[param];
        }
    }
    std::string newExperimentName = newExperimentNameSs.str();
    newExperimentName.append(it, end);
    experimentName = newExperimentName;

    return experimentName;
}

std::filesystem::path experimentRootFromString(const std::string& experimentRoot)
{
    std::filesystem::path root = experimentRoot;
    if (!root.is_absolute())
        root = doot2::experimentsDirectory / root;
    return root;
}

std::vector<nlohmann::json> flattenGridSearchParameters(const nlohmann::json& json)
{
    auto jsonTail = json;
    jsonTail.erase(json.begin().key());
    std::vector<nlohmann::json> tail;
    if (!jsonTail.empty())
        tail = flattenGridSearchParameters(jsonTail);

    std::vector<nlohmann::json> out;
    for (auto& v : json.begin().value()) {
        nlohmann::json j;
        j[json.begin().key()] = v;
        if (tail.empty()) {
            out.push_back(std::move(j));
        }
        else {
            for (auto& t: tail) {
                auto j2 = j;
                j2.update(t);
                out.push_back(std::move(j2));
            }
        }
    }
    return out;
}
