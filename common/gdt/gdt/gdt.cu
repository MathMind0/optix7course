// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include "gdt.h"
#include "math/LinearSpace.h"
#include "math/AffineSpace.h"

#ifdef _WIN32
// On Windows the console does not interpret ANSI escape sequences
// (GDT_TERMINAL_*) unless virtual-terminal processing is enabled; otherwise
// the ESC byte is printed literally (e.g. "?[1;32m"). Enable it once at
// process startup. This is host code and runs whenever the gdt library is
// linked into an executable.
namespace {
struct EnableConsoleColors {
    EnableConsoleColors() {
        HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
        if (hOut == INVALID_HANDLE_VALUE || hOut == NULL) return;
        DWORD mode = 0;
        if (!GetConsoleMode(hOut, &mode)) return;
        mode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
        SetConsoleMode(hOut, mode);
    }
};
static EnableConsoleColors g_enableConsoleColors;
}
#endif
