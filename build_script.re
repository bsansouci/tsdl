open Bsb_internals;

let ( +/ ) = Filename.concat;

gcc("lib" +/ "tsdl_new.o", ["src" +/ "tsdl_new.c"]);
