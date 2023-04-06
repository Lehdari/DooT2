//
// Project: DooT2
// File: SequenceStorage.inl
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

template<typename T_Data, typename T_DefaultEntry>
void SequenceStorage::addSequence(
    const std::string& sequenceName,
    const T_DefaultEntry& defaultValue,
    const std::vector<int64_t>& shape)
{
    auto& sequence = getSequenceWrapper<T_Data>(sequenceName, _batchSize, defaultValue, shape);
    sequence.resize(_size);
}

template<typename T_Data>
void SequenceStorage::addSequence(
    const std::string& sequenceName,
    const torch::Tensor& defaultValue)
{
    std::vector<int64_t> shape = defaultValue.sizes().vec();
    addSequence<T_Data, torch::Tensor>(sequenceName, defaultValue, shape);
}

template<typename T_Data>
void SequenceStorage::addEntry(const std::string& sequenceName, const T_Data& entry)
{
    auto& instanceStorage = storage[this];

    if (!instanceStorage.contains(sequenceName))
        throw std::runtime_error("No sequence with name \"" + sequenceName + "\"");

    for (auto& [name, sequence] : instanceStorage) {
        size_t sequenceSize = 0;
        if (name == sequenceName) {
            assert(sequence.getTypeId() == typeId<T_Data>()); // check the correct type
            sequenceSize = sequence.addEntry(&entry);
        }
        else {
            sequenceSize = sequence.addEntry(); // add default value to all other sequence except the one specified
        }
        assert(sequenceSize == _size+1); // indicates a bug in SequenceStorage implementation (some of the sequence has a differing length)
    }

    ++_size;
}

template<typename... T_NamesAndEntries>
void SequenceStorage::addEntries(const T_NamesAndEntries&... namesAndEntries)
{
    static_assert(sizeof...(T_NamesAndEntries) % 2 == 0); // check for even number of arguments
    addEntriesRecursive(namesAndEntries...);

    ++_size;

    // resize all sequence not specified in the arguments to match the new size
    auto& instanceStorage = storage[this];
    for (auto& [name, sequence] : instanceStorage) {
        sequence.resize(_size);
    }
}

template<typename T_Data>
const Sequence<T_Data>* SequenceStorage::getSequence(const std::string& sequenceName) const
{
    auto& instanceStorage = storage[this];
    if (!instanceStorage.contains(sequenceName))
        return nullptr; // no such sequence

    // check the type
    assert(instanceStorage.at(sequenceName).getTypeId() == typeId<T_Data>());

    return &instanceStorage.at(sequenceName).getSequence<T_Data>();
}

template<typename T_Data>
Sequence<T_Data>::BatchHandle SequenceStorage::getBatch(const std::string& sequenceName, size_t t)
{
    auto& instanceStorage = storage[this];
    if (!instanceStorage.contains(sequenceName))
        throw std::runtime_error("No sequence named \"" + sequenceName + "\"");

    // check the type
    assert(instanceStorage.at(sequenceName).getTypeId() == typeId<T_Data>());

    // check bounds
    assert(t < _size);

    return instanceStorage.at(sequenceName).getSequence<T_Data>()[t];
}

template<typename T_FirstEntry, typename... T_NamesAndEntries>
void SequenceStorage::addEntriesRecursive(
    const std::string& sequenceName,
    const T_FirstEntry& firstEntry,
    const T_NamesAndEntries&... entries)
{
    auto& instanceStorage = storage[this];

    if (!instanceStorage.contains(sequenceName))
        addSequenceWrapper(sequenceName, firstEntry);

    auto& sequence = instanceStorage.at(sequenceName);
    assert(sequence.getTypeId() == typeId<T_FirstEntry>()); // check the correct type

    size_t sequenceSize = sequence.addEntry(&firstEntry);
    assert(sequenceSize == _size+1); // indicates a bug

    // process rest of the entries
    addEntriesRecursive(entries...);
}


template <typename T_Data, typename T_DefaultEntry>
size_t SequenceStorage::SequenceWrapper::adder(const void* entry)
{
    assert(_dataTypeId == SequenceStorage::typeId<T_Data>());
    auto& data = *static_cast<Sequence<T_Data>*>(_data);
    if (entry == nullptr) {
        data.addBatch(*static_cast<const T_DefaultEntry*>(_defaultValue));
    }
    else {
        data.addBatch(*static_cast<const T_Data*>(entry));
    }
    return static_cast<Sequence<T_Data>*>(_data)->length();
}

template <typename T_Data>
size_t SequenceStorage::SequenceWrapper::adderTensor(const torch::Tensor& tensor)
{
    assert(_dataTypeId == SequenceStorage::typeId<T_Data>());
    if constexpr (isTensorScalarType<T_Data>()) {
        auto& data = *static_cast<Sequence<T_Data>*>(_data);
        data.addBatch(tensor);
        return static_cast<Sequence<T_Data>*>(_data)->length();
    }
    else
        (this->*_adder)(nullptr); // Add default value
}

template<typename T_Data, typename T_DefaultEntry>
void SequenceStorage::SequenceWrapper::deleter()
{
    assert(_dataTypeId == SequenceStorage::typeId<T_Data>());
    delete static_cast<Sequence<T_Data>*>(_data);
    delete static_cast<const T_DefaultEntry*>(_defaultValue);
}

template<typename T_Data>
void* SequenceStorage::SequenceWrapper::dataCopier() const
{
    assert(_dataTypeId == SequenceStorage::typeId<T_Data>());
    // copy construct a new data vector
    return new Sequence<T_Data>(*static_cast<Sequence<T_Data>*>(_data));
}

template<typename T_DefaultEntry>
const void* SequenceStorage::SequenceWrapper::defaultCopier() const
{
    assert(_defaultValueTypeId == SequenceStorage::typeId<T_DefaultEntry>());
    // copy construct a new default type
    return new T_DefaultEntry(*static_cast<const T_DefaultEntry*>(_defaultValue));
}

template<typename T_Data, typename T_DefaultEntry>
void SequenceStorage::SequenceWrapper::resizer(size_t size)
{
    assert(_dataTypeId == SequenceStorage::typeId<T_Data>());
    static_cast<Sequence<T_Data>*>(_data)->resize(
        size, *static_cast<const T_DefaultEntry*>(_defaultValue));
}

template<typename T_Data, typename T_DefaultEntry>
SequenceStorage::SequenceWrapper::SequenceWrapper(
    T_Data* dummy,
    int64_t batchSize,
    const T_DefaultEntry& defaultValue,
    const std::vector<int64_t>& shape
) :
    _data               (new Sequence<T_Data>(batchSize, shape)),
    _defaultValue       (new T_DefaultEntry(defaultValue)),
    _dataTypeId         (SequenceStorage::typeId<T_Data>()),
    _defaultValueTypeId (SequenceStorage::typeId<T_DefaultEntry>()),
    _adder              (&SequenceStorage::SequenceWrapper::adder<T_Data, T_DefaultEntry>),
    _adderTensor        (&SequenceStorage::SequenceWrapper::adderTensor<T_Data>),
    _deleter            (&SequenceStorage::SequenceWrapper::deleter<T_Data, T_DefaultEntry>),
    _dataCopier         (&SequenceStorage::SequenceWrapper::dataCopier<T_Data>),
    _defaultCopier      (&SequenceStorage::SequenceWrapper::defaultCopier<T_DefaultEntry>),
    _resizer            (&SequenceStorage::SequenceWrapper::resizer<T_Data, T_DefaultEntry>)
{
}

template<typename T_Data>
Sequence<T_Data>& SequenceStorage::SequenceWrapper::getSequence()
{
    auto newTypeId = SequenceStorage::typeId<T_Data>();
    assert(_dataTypeId == newTypeId);
    return *static_cast<Sequence<T_Data>*>(_data);
}

template<typename T_Data, typename T_DefaultEntry>
SequenceStorage::SequenceWrapper& SequenceStorage::getSequenceWrapper(
    const std::string& sequenceName,
    int64_t batchSize,
    const T_DefaultEntry& defaultValue,
    const std::vector<int64_t>& shape
) {
    // storage dedicated for this instance
    auto& instanceStorage = storage[this];
    // checking of the existing entry needs to be done manually due to the templated
    // constructor of SequenceWrapper
    if (!instanceStorage.contains(sequenceName))
        instanceStorage.emplace(std::make_pair(sequenceName,
            SequenceWrapper((T_Data*)nullptr, batchSize, defaultValue, shape)));
    return instanceStorage.at(sequenceName);
}

template<typename T_Data>
uint32_t SequenceStorage::typeId()
{
    static uint32_t typeId {typeIdCounter++};
    return typeId;
}
