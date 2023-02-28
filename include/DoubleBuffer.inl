//
// Project: DooT2
// File: DoubleBuffer.inl
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

template<typename T_Data>
DoubleBuffer<T_Data>::ReadHandle::~ReadHandle()
{
    *_readHandleActive = false;
    _lock.unlock();
    _readHandleCV->notify_one();
}

template<typename T_Data>
DoubleBuffer<T_Data>::ReadHandle::ReadHandle(DoubleBuffer<T_Data>& doubleBuffer) :
    _data               (nullptr),
    _readHandleActive   (&doubleBuffer._readHandleActive),
    _readHandleCV       (&doubleBuffer._readHandleCV),
    _lock               (doubleBuffer._readHandleMutex)
{
    doubleBuffer._readHandleCV.wait(_lock, [&]{ return !(*_readHandleActive); });
    _data = doubleBuffer._read;
    *_readHandleActive = true;
}

template<typename T_Data>
const T_Data& DoubleBuffer<T_Data>::ReadHandle::operator*() const noexcept
{
    return *_data;
}

template<typename T_Data>
DoubleBuffer<T_Data>::WriteHandle::~WriteHandle()
{
    *_writeHandleActive = false;
    *_updated = true;
    _lock.unlock();
    _writeHandleCV->notify_one();
}

template<typename T_Data>
DoubleBuffer<T_Data>::WriteHandle::WriteHandle(DoubleBuffer<T_Data>& doubleBuffer) :
    _data               (nullptr),
    _updated            (&doubleBuffer._updated),
    _writeHandleActive  (&doubleBuffer._writeHandleActive),
    _writeHandleCV      (&doubleBuffer._writeHandleCV),
    _lock               (doubleBuffer._writeHandleMutex)
{
    doubleBuffer._writeHandleCV.wait(_lock, [&]{ return !(*_writeHandleActive); });
    _data = doubleBuffer._write;
    *_writeHandleActive = true;
}

template<typename T_Data>
T_Data& DoubleBuffer<T_Data>::WriteHandle::operator*() noexcept
{
    return *_data;
}

template<typename T_Data>
DoubleBuffer<T_Data>::DoubleBuffer(const T_Data& data) :
    _data1              (data),
    _data2              (data),
    _read               (&_data2),
    _write              (&_data1),
    _readHandleActive   (false),
    _writeHandleActive  (false)
{
}

template<typename T_Data>
const DoubleBuffer<T_Data>::ReadHandle DoubleBuffer<T_Data>::read()
{
    // Perform swap if the buffer has been written into
    if (_updated) {
        std::unique_lock<std::mutex> writeLock(_writeHandleMutex);
        _writeHandleCV.wait(writeLock, [&]{ return !_writeHandleActive; });
        _writeHandleActive = true;

        std::unique_lock<std::mutex> readLock(_readHandleMutex);
        _readHandleCV.wait(readLock, [&]{ return !_readHandleActive; });
        _readHandleActive = true;

        std::swap(_read, _write);

        _readHandleActive = false;
        _writeHandleActive = false;
        _updated = false;

        writeLock.unlock();
        readLock.unlock();

        _writeHandleCV.notify_one();
        _readHandleCV.notify_one();
    }

    return ReadHandle(*this);
}

template<typename T_Data>
DoubleBuffer<T_Data>::WriteHandle DoubleBuffer<T_Data>::write()
{
    return WriteHandle(*this);
}
