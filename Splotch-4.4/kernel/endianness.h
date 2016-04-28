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

#ifndef RAYPP_ENDIANNESS_H
#define RAYPP_ENDIANNESS_H

#include "cxxsupport/datatypes.h"

class EndianTest__
  {
  private:
    bool big_end;

  public:
    EndianTest__()
      {
      union { uint16 i16; uint8 i8; } tmp;
      tmp.i16 = 1;
      big_end = (tmp.i8==0);
      }
    operator bool() const { return big_end; }
  };

const EndianTest__ big_endian;

#endif
