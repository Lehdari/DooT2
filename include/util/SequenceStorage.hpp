//
// Project: DooT2
// File: SequenceStorage.hpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once

#include "util/Sequence.hpp"

#include <vector>
#include <unordered_map>
#include <cassert>


class SequenceStorage {
public:
    SequenceStorage(int64_t batchSize);
    ~SequenceStorage();
    SequenceStorage(const SequenceStorage& other);
    SequenceStorage(SequenceStorage&& other) noexcept;
    SequenceStorage& operator=(const SequenceStorage& other);
    SequenceStorage& operator=(SequenceStorage&& other) noexcept;

    // Add new sequence, initializes the sequence vector to size of the sequence (.size()),
    // filled with defaultValue.
    template <typename T_Data, typename T_DefaultEntry>
    void addSequence(
        const std::string& sequenceName,
        const T_DefaultEntry& defaultValue,
        const std::vector<int64_t>& shape = {1});
    // Same as above but shape is deduced from the tensor
    template <typename T_Data>
    void addSequence(
        const std::string& sequenceName,
        const torch::Tensor& defaultValue);

    // Add entry to a sequence (similar to push_back), if one with specified name does not exist,
    // std::runtime_error gets thrown
    // Note: All sequence in a SequenceStorage instance will always have equal length, meaning
    //       an entry will be added to each sequence vector using their respective defaultValues.
    //       In case this is undesired, see addEntries
    template <typename T_Data>
    void addEntry(const std::string& sequenceName, const T_Data& entry);
    void addEntry(const std::string& sequenceName, const torch::Tensor& entry);

    // Add multiple entries, arguments must be alternating sequence of std::string (sequence name)
    // followed T_Data (type of the sequence). For example:
    // addEntries("timestamp", 83745622, "value", 0.2533);
    // This function also adds new sequence in case the ones with the names specified do not exist
    template <typename... T_NamesAndEntries>
    void addEntries(const T_NamesAndEntries&... namesAndEntries);

    void resize(size_t newLength);

    // Get pointer to a sequence. Returns nullptr in case a sequence with
    // name sequenceName does not exist.
    template <typename T_Data>
    const Sequence<T_Data>* getSequence(const std::string& sequenceName) const;

    // Get handle to a batch at step t
    template <typename T_Data>
    Sequence<T_Data>::BatchHandle getBatch(const std::string& sequenceName, size_t t);

    // List all sequence names
    std::vector<std::string> getSequenceNames() const;

    // Get number of sequences (matches length of the vector returned from getSequenceNames)
    size_t getNumSequences() const noexcept;

    // Get SequenceStorage length (size() of sequence vectors)
    size_t length() const noexcept;

    int64_t batchSize() const noexcept;

    // Get total number of SequenceStorage instances
    static size_t getNumInstances();

private:
    size_t  _size;
    int64_t _batchSize;

    template<typename T_FirstEntry, typename... T_NamesAndEntries>
    void addEntriesRecursive(const std::string& sequenceName, const T_FirstEntry& firstEntry,
        const T_NamesAndEntries&... namesAndEntries);

    void addEntriesRecursive();

    // A simple wrapper for sequence data vector that provides deletion / copying
    // functionality with type hiding
    class SequenceWrapper {
        using Adder         = size_t (SequenceWrapper::*)(const void*);
        using AdderTensor   = size_t (SequenceWrapper::*)(const torch::Tensor&);
        using Deleter       = void (SequenceWrapper::*)();
        using DataCopier    = void* (SequenceWrapper::*)() const;
        using DefaultCopier = const void* (SequenceWrapper::*)() const;
        using Resizer       = void (SequenceWrapper::*)(size_t);

        void*           _data;                  // the sequence data vector, pointer to std::vector<T_Data>
        const void*     _defaultValue;          // default value, pointer to T_DefaultEntry
        uint32_t        _dataTypeId;            // type ID for safety checking
        uint32_t        _defaultValueTypeId;    // type ID for safety checking
        Adder           _adder;                 // adder function pointer that remembers the T_Data type
        AdderTensor     _adderTensor;           // adder function for tensors that remembers the T_Data type
        Deleter         _deleter;               // deleter function pointer that remembers the T_Data type
        DataCopier      _dataCopier;            // data copier function pointer that remembers the T_Data type
        DefaultCopier   _defaultCopier;         // default value copier function pointer that remembers the T_Data type
        Resizer         _resizer;               // data vector resizer function pointer that remembers the T_Data type

        // Function used to store in _deleter function pointers
        template <typename T_Data, typename T_DefaultEntry>
        size_t adder(const void* entry);

        template <typename T_Data>
        size_t adderTensor(const torch::Tensor& tensor);

        // Function used to store in _deleter function pointers
        template<typename T_Data, typename T_DefaultEntry>
        void deleter();

        // Function used to store in _dataCopier function pointers
        template <typename T_Data>
        void* dataCopier() const;

        // Function used to store in _defaultCopier function pointers
        template <typename T_DefaultEntry>
        const void* defaultCopier() const;

        // Function used to store in _resizer function pointers
        template <typename T_Data, typename T_DefaultEntry>
        void resizer(size_t size);

    public:
        template <typename T_Data, typename T_DefaultEntry>
        SequenceWrapper(
            T_Data* dummy,
            int64_t batchSize,
            const T_DefaultEntry& defaultValue,
            const std::vector<int64_t>& shape);
        SequenceWrapper(const SequenceWrapper& other);
        SequenceWrapper(SequenceWrapper&& other) noexcept;
        SequenceWrapper& operator=(const SequenceWrapper& other);
        SequenceWrapper& operator=(SequenceWrapper&& other) noexcept;
        ~SequenceWrapper();

        // Add entry, returns the new sequence vector size
        size_t addEntry(const void* entry = nullptr); // nullptr: use the stored default value
        size_t addEntry(const torch::Tensor& entry); // nullptr: use the stored default value
        void resize(size_t size);
        uint32_t getTypeId() const noexcept;

        template <typename T_Data>
        Sequence<T_Data>& getSequence();
    };

    // Map from SequenceStorage instances (or rather pointers to them) to vectors
    // containing all the typed data. The actual type handling is done inside the
    // utility functions below.
    // SequenceStorage ptr -> sequence name -> type -> sequence data
    using Storage = std::unordered_map<const SequenceStorage*, std::unordered_map<std::string, SequenceWrapper>>;

    static Storage  storage;
    static uint32_t typeIdCounter;

    // Access the underlying sequence container, create one if necessary
    template <typename T_Data, typename T_DefaultEntry>
    SequenceWrapper& getSequenceWrapper(
        const std::string& sequenceName,
        int64_t batchSize,
        const T_DefaultEntry& defaultValue,
        const std::vector<int64_t>& shape);

    // Function for generating distinct IDs for each T_Data type
    template <typename T_Data>
    static uint32_t typeId();
};


#include "SequenceStorage.inl"
