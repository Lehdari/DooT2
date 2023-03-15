//
// Project: DooT2
// File: TimeSeries.hpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once

#include <nlohmann/json.hpp>
#include <vector>
#include <unordered_map>
#include <cassert>


class TimeSeries {
public:
    TimeSeries();
    ~TimeSeries();
    TimeSeries(const TimeSeries& other);
    TimeSeries(TimeSeries&& other) noexcept;
    TimeSeries& operator=(const TimeSeries& other);
    TimeSeries& operator=(TimeSeries&& other) noexcept;

    // Add new series, initializes the series vector to size of the time series (.size()),
    // filled with defaultValue.
    template <typename T_Entry>
    void addSeries(const std::string& seriesName, const T_Entry& defaultValue = T_Entry());

    // Add entry to a series (similar to push_back), if one with specified name does not exist,
    // addSeries gets called with seriesName=seriesName and defaultValue=entry.
    // Note: All series in a TimeSeries instance will always have equal length, meaning
    //       an entry will be added to each series vector using their respective defaultValues.
    //       In case this is undesired, see addEntries
    template <typename T_Entry>
    void addEntry(const std::string& seriesName, const T_Entry& entry);

    // Add multiple entries, arguments must be alternating sequence of std::string (series name)
    // followed T_Entry (type of the series). For example:
    // addEntries("timestamp", 83745622, "value", 0.2533);
    // This function also adds new series in case the ones with the names specified do not exist
    template <typename... T_NamesAndEntries>
    void addEntries(const T_NamesAndEntries&... namesAndEntries);

    // Get pointer to a data vector of a time series. Returns nullptr in case a series with
    // name seriesName does not exist.
    template <typename T_Entry>
    const std::vector<T_Entry>* getSeriesVector(const std::string& seriesName) const;

    // List all series names
    std::vector<std::string> getSeriesNames() const;

    // Get number of series (matches length of the vector returned from getSeriesNames)
    size_t getNumSeries() const noexcept;

    // Get TimeSeries length (size() of series vectors)
    size_t length();

    // Serialize to JSON
    nlohmann::json toJson() const;

    // Get total number of TimeSeries instances
    static size_t getNumInstances();

private:
    size_t  _size;

    template<typename T_FirstEntry, typename... T_NamesAndEntries>
    void addEntriesRecursive(const std::string& seriesName, const T_FirstEntry& firstEntry,
        const T_NamesAndEntries&... namesAndEntries);

    void addEntriesRecursive();

    // A simple container for series data vector that provides deletion / copying
    // functionality with type hiding
    class Series {
        using Adder         = size_t (Series::*)(const void*);
        using Deleter       = void (Series::*)();
        using DataCopier    = void* (Series::*)() const;
        using DefaultCopier = const void* (Series::*)() const;
        using Resizer       = void (Series::*)(size_t);
        using Serializer    = void (Series::*)(const std::string&, nlohmann::json&) const;

        void*           _data;          // the series data vector, pointer to std::vector<T_Entry>
        const void*     _defaultValue;  // default value, pointer to T_Entry
        uint32_t        _typeId;        // type ID for safety checking
        Adder           _adder;         // adder function pointer that remembers the T_Entry type
        Deleter         _deleter;       // deleter function pointer that remembers the T_Entry type
        DataCopier      _dataCopier;    // data copier function pointer that remembers the T_Entry type
        DefaultCopier   _defaultCopier; // default value copier function pointer that remembers the T_Entry type
        Resizer         _resizer;       // data vector resizer function pointer that remembers the T_Entry type
        Serializer      _serializer;       // data vector resizer function pointer that remembers the T_Entry type

        // Function used to store in _deleter function pointers
        template <typename T_Entry>
        size_t adder(const void* entry);

        // Function used to store in _deleter function pointers
        template <typename T_Entry>
        void deleter();

        // Function used to store in _dataCopier function pointers
        template <typename T_Entry>
        void* dataCopier() const;

        // Function used to store in _defaultCopier function pointers
        template <typename T_Entry>
        const void* defaultCopier() const;

        // Function used to store in _resizer function pointers
        template <typename T_Entry>
        void resizer(size_t size);

        // Function used to store in _serializer function pointers
        template <typename T_Entry>
        void serializer(const std::string& seriesName, nlohmann::json& json) const;

    public:
        template <typename T_Entry>
        Series(const T_Entry& defaultValue);
        Series(const Series& other);
        Series(Series&& other) = default;
        Series& operator=(const Series& other);
        Series& operator=(Series&& other) = default;
        ~Series();

        // Add entry, returns the new series vector size
        size_t addEntry(const void* entry = nullptr); // nullptr: use the stored default value
        void resize(size_t size);
        void serialize(const std::string& seriesName, nlohmann::json& json) const;
        uint32_t getTypeId() const noexcept;

        template <typename T_Entry>
        const std::vector<T_Entry>& getVector();
    };

    // Map from TimeSeries instances (or rather pointers to them) to vectors
    // containing all the typed data. The actual type handling is done inside the
    // utility functions below.
    // TimeSeries ptr -> series name -> type -> series data
    using Storage = std::unordered_map<const TimeSeries*, std::unordered_map<std::string, Series>>;

    static Storage  storage;
    static uint32_t typeIdCounter;

    // Access the underlying series container, create one if necessary
    template <typename T_Entry>
    Series& getSeries(const std::string& seriesName, const T_Entry& defaultValue);

    // Function for generating distinct IDs for each T_Entry type
    template <typename T_Entry>
    static uint32_t typeId();
};


#include "TimeSeries.inl"
