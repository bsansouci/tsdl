open Bsb_internals;

let ( +/ ) = Filename.concat;

gcc(~includes=[".." +/ "sdl2" +/ "include"], "lib" +/ "tsdl_new.o", ["src" +/ "tsdl_new.c"]);
