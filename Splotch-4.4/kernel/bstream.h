/*
 *  Ray++ - Object-oriented ray tracing library
 *  Copyright (C) 1998-2004 Martin Reinecke and others.
 *  See the AUTHORS file for more information.
 *
 *  This library is free software; you can redistribute it and/or
 *  modify it under the terms of the GNU Library General Public
 *  License as published by the Free Software Foundation; either
 *  version 2 of the License, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *  Library General Public License for more details.
 *
 *  You should have received a copy of the GNU Library General Public
 *  License along with this library; if not, write to the Free
 *  Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 *
 *  See the README file for more information.
 */

#ifndef RAYPP_BSTREAM_H
#define RAYPP_BSTREAM_H

#include <fstream>
#include "kernel/byteswap.h"
#include "kernel/endianness.h"

const bool file_is_lsb=big_endian, file_is_msb=!big_endian,
           file_is_natural=false;

class bofstream: public std::ofstream
  {
  private:
    bool doswap;

  public:
    /*! */
    bofstream (const char *fname, bool doswap_)
      : std::ofstream(fname,std::ios::binary), doswap(doswap_) {}

    template<typename T> bofstream &operator<< (const T &data)
      {
      if (doswap)
        {
        T tmp = data;
        byteswap (tmp);
        write (reinterpret_cast<const char *> (&tmp), sizeof(T));
        }
      else
        write (reinterpret_cast<const char *> (&data), sizeof(T));
      return *this;
      }
    template<typename T> bofstream &put (const T *data, size_t num)
      {
      if (doswap)
        for (size_t m=0; m<num; ++m)
          {
          T tmp=data[m];
          byteswap (tmp);
          write (reinterpret_cast<const char *> (&tmp), sizeof(T));
          }
      else
        write (reinterpret_cast<const char *> (data), num*sizeof(T));
      return *this;
      }
  };

class bifstream: public std::ifstream
  {
  private:
    bool doswap;

  public:
    /*! */
    bifstream ()
      : doswap(false) {}
    bifstream (const char *fname, bool doswap_)
      : std::ifstream(fname,std::ios::binary), doswap(doswap_) {}

    void open (const char *fname, bool doswap_)
      {
      doswap=doswap_;
      std::ifstream::open(fname,std::ios::binary);
      }

    template<typename T> bifstream &operator>> (T &data)
      {
      read (reinterpret_cast<char *> (&data), sizeof(T));
      if (doswap) byteswap (data);
      return *this;
      }
    template<typename T> bifstream &get (T *data, size_t num)
      {
      read (reinterpret_cast<char *> (data), num*sizeof(T));
      if (doswap)
        for (size_t m=0; m<num; ++m)
          byteswap (data[m]);
      return *this;
      }

    void rewind()
      { seekg(0,std::ios::beg); }
    void skip(std::streamoff nbytes)
      { seekg(nbytes,std::ios::cur); }
  };

#endif
