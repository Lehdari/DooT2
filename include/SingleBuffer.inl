//
// Project: DooT2
// File: SingleBuffer.inl
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

template<typename T_Data>
SingleBuffer<T_Data>::ReadHandle::~ReadHandle()
{
    *_handleActive = false;
    _lock.unlock();
    _handleCV->notify_one();
}

template<typename T_Data>
SingleBuffer<T_Data>::ReadHandle::ReadHandle(SingleBuffer<T_Data>& singleBuffer) :
    _data           (nullptr),
    _handleActive   (&singleBuffer._handleActive),
    _handleCV       (&singleBuffer._handleCV),
    _lock           (singleBuffer._handleMutex)
{
    singleBuffer._handleCV.wait(_lock, [&]{ return !(*_handleActive); });
    _data = &singleBuffer._data;
    *_handleActive = true;
}

template<typename T_Data>
const T_Data& SingleBuffer<T_Data>::ReadHandle::operator*() const noexcept
{
    return *_data;
}

template<typename T_Data>
const T_Data* SingleBuffer<T_Data>::ReadHandle::operator->() const noexcept
{
    return _data;
}

template<typename T_Data>
const T_Data* SingleBuffer<T_Data>::ReadHandle::get() const noexcept
{
    return _data;
}

template<typename T_Data>
SingleBuffer<T_Data>::WriteHandle::~WriteHandle()
{
    *_handleActive = false;
    _lock.unlock();
    _handleCV->notify_one();
}

template<typename T_Data>
SingleBuffer<T_Data>::WriteHandle::WriteHandle(SingleBuffer<T_Data>& singleBuffer) :
    _data           (nullptr),
    _handleActive   (&singleBuffer._handleActive),
    _handleCV       (&singleBuffer._handleCV),
    _lock           (singleBuffer._handleMutex)
{
    singleBuffer._handleCV.wait(_lock, [&]{ return !(*_handleActive); });
    _data = &singleBuffer._data;
    *_handleActive = true;
}

template<typename T_Data>
T_Data& SingleBuffer<T_Data>::WriteHandle::operator*() noexcept
{
    return *_data;
}

template<typename T_Data>
T_Data* SingleBuffer<T_Data>::WriteHandle::operator->() noexcept
{
    return _data;
}

template<typename T_Data>
T_Data* SingleBuffer<T_Data>::WriteHandle::get() noexcept
{
    return _data;
}

template<typename T_Data>
SingleBuffer<T_Data>::SingleBuffer(const T_Data& data) :
    _data           (data),
    _handleActive   (false)
{
}

template<typename T_Data>
const SingleBuffer<T_Data>::ReadHandle SingleBuffer<T_Data>::read()
{
    return ReadHandle(*this);
}

template<typename T_Data>
SingleBuffer<T_Data>::WriteHandle SingleBuffer<T_Data>::write()
{
    return WriteHandle(*this);
}
