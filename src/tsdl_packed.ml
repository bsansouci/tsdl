module Unsigned : sig 
#1 "unsigned.mli"
(*
 * Copyright (c) 2013 Jeremy Yallop.
 *
 * This file is distributed under the terms of the MIT License.
 * See the file LICENSE for details.
 *)

(** Types and operations for unsigned integers. *)

module type S = sig
  type t

  val add : t -> t -> t
  (** Addition. *)

  val sub : t -> t -> t
  (** Subtraction. *)

  val mul : t -> t -> t
  (** Multiplication. *)

  val div : t -> t -> t
  (** Division.  Raise {!Division_by_zero} if the second argument is zero. *)

  val rem : t -> t -> t
  (** Integer remainder.  Raise {!Division_by_zero} if the second argument is
      zero. *)

  val max_int : t
  (** The greatest representable integer. *)

  val logand : t -> t -> t
  (** Bitwise logical and. *)

  val logor : t -> t -> t
  (** Bitwise logical or. *)

  val logxor : t -> t -> t
  (** Bitwise logical exclusive or. *)

  val shift_left : t -> int -> t
  (** {!shift_left} [x] [y] shifts [x] to the left by [y] bits. *)

  val shift_right : t -> int -> t
  (** {!shift_right} [x] [y] shifts [x] to the right by [y] bits. *)

  val of_int : int -> t
  (** Convert the given int value to an unsigned integer. *)

  val to_int : t -> int
  (** Convert the given unsigned integer value to an int. *)

  val of_int64 : int64 -> t
  (** Convert the given int64 value to an unsigned integer. *)

  val to_int64 : t -> int64
  (** Convert the given unsigned integer value to an int64. *)

  val of_string : string -> t
  (** Convert the given string to an unsigned integer.  Raise {!Failure}
      ["int_of_string"] if the given string is not a valid representation of
      an unsigned integer. *)

  val to_string : t -> string
  (** Return the string representation of its argument. *)

  val zero : t
  (** The integer 0. *)

  val one : t
  (** The integer 1. *)

  val lognot : t -> t
  (** Bitwise logical negation. *)

  val succ : t -> t
  (** Successor. *)

  val pred : t -> t
  (** Predecessor. *)

  val compare : t -> t -> int
  (** The comparison function for unsigned integers, with the same
      specification as {!Pervasives.compare}. *)

  module Infix : sig
    val (+) : t -> t -> t
    (** Addition.  See {!add}. *)

    val (-) : t -> t -> t
    (** Subtraction.  See {!sub}.*)

    val ( * ) : t -> t -> t
    (** Multiplication.  See {!mul}.*)

    val (/) : t -> t -> t
    (** Division.  See {!div}.*)

    val (mod) : t -> t -> t
    (** Integer remainder.  See {!rem}. *)

    val (land) : t -> t -> t
    (** Bitwise logical and.  See {!logand}. *)

    val (lor) : t -> t -> t
    (** Bitwise logical or.  See {!logor}. *)

    val (lxor) : t -> t -> t
    (** Bitwise logical exclusive or.  See {!logxor}. *)

    val (lsl) : t -> int -> t
    (** [x lsl y] shifts [x] to the left by [y] bits.  See {!shift_left}. *)

    val (lsr) : t -> int -> t
    (** [x lsr y] shifts [x] to the right by [y] bits.  See {!shift_right}. *)
  end
(** Infix names for the unsigned integer operations. *)
end
(** Unsigned integer operations. *)

module UChar : S
(** Unsigned char type and operations. *)

module UInt8 : S
(** Unsigned 8-bit integer type and operations. *)

module UInt16 : S
(** Unsigned 16-bit integer type and operations. *)

module UInt32 : sig
  include S
  val of_int32 : int32 -> t
  val to_int32 : t -> int32
end
(** Unsigned 32-bit integer type and operations. *)

module UInt64 : sig
  include S
  val of_int64 : int64 -> t
  val to_int64 : t -> int64
end
(** Unsigned 64-bit integer type and operations. *)

module Size_t : S
(** The size_t unsigned integer type and operations. *)

module UShort : S
(** The unsigned short integer type and operations. *)

module UInt : S
(** The unsigned int type and operations. *)

module ULong : S
(** The unsigned long integer type and operations. *)

module ULLong : S
(** The unsigned long long integer type and operations. *)


type uchar = UChar.t
(** The unsigned char type. *)

type uint8 = UInt8.t
(** Unsigned 8-bit integer type. *)

type uint16 = UInt16.t
(** Unsigned 16-bit integer type. *)

type uint32 = UInt32.t
(** Unsigned 32-bit integer type. *)

type uint64 = UInt64.t
(** Unsigned 64-bit integer type. *)

type size_t = Size_t.t
(** The size_t unsigned integer type. *)

type ushort = UShort.t
(** The unsigned short unsigned integer type. *)

type uint = UInt.t
(** The unsigned int type. *)

type ulong = ULong.t
(** The unsigned long integer type. *)

type ullong = ULLong.t
(** The unsigned long long integer type. *)

end = struct
#1 "unsigned.ml"
(*
 * Copyright (c) 2013 Jeremy Yallop.
 *
 * This file is distributed under the terms of the MIT License.
 * See the file LICENSE for details.
 *)

(* Boxed unsigned types *)
module type Basics = sig
  type t

  val add : t -> t -> t
  val sub : t -> t -> t
  val mul : t -> t -> t
  val div : t -> t -> t
  val rem : t -> t -> t
  val max_int : t
  val logand : t -> t -> t
  val logor : t -> t -> t
  val logxor : t -> t -> t
  val shift_left : t -> int -> t
  val shift_right : t -> int -> t
  val of_int : int -> t
  val to_int : t -> int
  val of_int64 : int64 -> t
  val to_int64 : t -> int64
  val of_string : string -> t
  val to_string : t -> string
end


module type Extras = sig
  type t

  val zero : t
  val one : t
  val lognot : t -> t
  val succ : t -> t
  val pred : t -> t
  val compare : t -> t -> int
end


module type Infix = sig
  type t
  val (+) : t -> t -> t
  val (-) : t -> t -> t
  val ( * ) : t -> t -> t
  val (/) : t -> t -> t
  val (mod) : t -> t -> t
  val (land) : t -> t -> t
  val (lor) : t -> t -> t
  val (lxor) : t -> t -> t
  val (lsl) : t -> int -> t
  val (lsr) : t -> int -> t
end


module type S = sig
  include Basics
  include Extras with type t := t

  module Infix : Infix with type t := t
end


module MakeInfix (B : Basics) =
struct
  open B
  let (+) = add
  let (-) = sub
  let ( * ) = mul
  let (/) = div
  let (mod) = rem
  let (land) = logand
  let (lor) = logor
  let (lxor) = logxor
  let (lsl) = shift_left
  let (lsr) = shift_right
end


module Extras(Basics : Basics) : Extras with type t := Basics.t =
struct
  open Basics
  let zero = of_int 0
  let one = of_int 1
  let succ n = add n one
  let pred n = sub n one
  let lognot n = logxor n max_int
  let compare (x : t) (y : t) = Pervasives.compare x y
end


module UInt8 : S =
struct
  module B =
  struct
    (* Once 4.01 support is dropped all of these should be [@@inline] *)
    type t = int
    let max_int = 255
    let add : t -> t -> t = fun x y -> (x + y) land max_int
    let sub : t -> t -> t = fun x y -> (x - y) land max_int
    let mul : t -> t -> t = fun x y -> (x * y) land max_int
    let div : t -> t -> t = (/)
    let rem : t -> t -> t = (mod)
    let logand: t -> t -> t = (land)
    let logor: t -> t -> t = (lor)
    let logxor : t -> t -> t = (lxor)
    let shift_left : t -> int -> t = fun x y -> (x lsl y) land max_int
    let shift_right : t -> int -> t = (lsr)
    let of_int (x: int): t =
      (* For backwards compatibility, this wraps *)
      x land max_int
    external to_int : t -> int = "%identity"
    let of_int64 : int64 -> t = fun x -> of_int (Int64.to_int x)
    let to_int64 : t -> int64 = fun x -> Int64.of_int (to_int x)
    external of_string : string -> t = "ctypes_uint8_of_string"
    let to_string : t -> string = string_of_int
  end
  include B
  include Extras(B)
  module Infix = MakeInfix(B)
end


module UInt16 : S =
struct
  module B =
  struct
    (* Once 4.01 support is dropped all of these should be [@@inline] *)
    type t = int
    let max_int = 65535
    let add : t -> t -> t = fun x y -> (x + y) land max_int
    let sub : t -> t -> t = fun x y -> (x - y) land max_int
    let mul : t -> t -> t = fun x y -> (x * y) land max_int
    let div : t -> t -> t = (/)
    let rem : t -> t -> t = (mod)
    let logand: t -> t -> t = (land)
    let logor: t -> t -> t = (lor)
    let logxor : t -> t -> t = (lxor)
    let shift_left : t -> int -> t = fun x y -> (x lsl y) land max_int
    let shift_right : t -> int -> t = (lsr)
    let of_int (x: int): t =
      (* For backwards compatibility, this wraps *)
      x land max_int
    external to_int : t -> int = "%identity"
    let of_int64 : int64 -> t = fun x -> Int64.to_int x |> of_int
    let to_int64 : t -> int64 = fun x -> to_int x |> Int64.of_int
    external of_string : string -> t = "ctypes_uint16_of_string"
    let to_string : t -> string = string_of_int
  end
  include B
  include Extras(B)
  module Infix = MakeInfix(B)
end


module UInt32 : sig
  include S
  external of_int32 : int32 -> t = "ctypes_uint32_of_int32"
  external to_int32 : t -> int32 = "ctypes_int32_of_uint32"
end =
struct
  module B =
  struct
    type t
    external add : t -> t -> t = "ctypes_uint32_add"
    external sub : t -> t -> t = "ctypes_uint32_sub"
    external mul : t -> t -> t = "ctypes_uint32_mul"
    external div : t -> t -> t = "ctypes_uint32_div"
    external rem : t -> t -> t = "ctypes_uint32_rem"
    external logand : t -> t -> t = "ctypes_uint32_logand"
    external logor : t -> t -> t = "ctypes_uint32_logor"
    external logxor : t -> t -> t = "ctypes_uint32_logxor"
    external shift_left : t -> int -> t = "ctypes_uint32_shift_left"
    external shift_right : t -> int -> t = "ctypes_uint32_shift_right"
    external of_int : int -> t = "ctypes_uint32_of_int"
    external to_int : t -> int = "ctypes_uint32_to_int"
    external of_int64 : int64 -> t = "ctypes_uint32_of_int64"
    external to_int64 : t -> int64 = "ctypes_uint32_to_int64"
    external of_string : string -> t = "ctypes_uint32_of_string"
    external to_string : t -> string = "ctypes_uint32_to_string"
    external _max_int : unit -> t = "ctypes_uint32_max"
    let max_int = _max_int ()
  end
  include B
  include Extras(B)
  module Infix = MakeInfix(B)
  external of_int32 : int32 -> t = "ctypes_uint32_of_int32"
  external to_int32 : t -> int32 = "ctypes_int32_of_uint32"
end


module UInt64 : sig
  include S
  external of_int64 : int64 -> t = "ctypes_uint64_of_int64"
  external to_int64 : t -> int64 = "ctypes_uint64_to_int64"
end =
struct
  module B =
  struct
    type t
    external add : t -> t -> t = "ctypes_uint64_add"
    external sub : t -> t -> t = "ctypes_uint64_sub"
    external mul : t -> t -> t = "ctypes_uint64_mul"
    external div : t -> t -> t = "ctypes_uint64_div"
    external rem : t -> t -> t = "ctypes_uint64_rem"
    external logand : t -> t -> t = "ctypes_uint64_logand"
    external logor : t -> t -> t = "ctypes_uint64_logor"
    external logxor : t -> t -> t = "ctypes_uint64_logxor"
    external shift_left : t -> int -> t = "ctypes_uint64_shift_left"
    external shift_right : t -> int -> t = "ctypes_uint64_shift_right"
    external of_int : int -> t = "ctypes_uint64_of_int"
    external to_int : t -> int = "ctypes_uint64_to_int"
    external of_int64 : int64 -> t = "ctypes_uint64_of_int64"
    external to_int64 : t -> int64 = "ctypes_uint64_to_int64"
    external of_string : string -> t = "ctypes_uint64_of_string"
    external to_string : t -> string = "ctypes_uint64_to_string"
    external _max_int : unit -> t = "ctypes_uint64_max"
    let max_int = _max_int ()
  end
  include B
  include Extras(B)
  module Infix = MakeInfix(B)
end


let pick : size:int -> (module S) =
  fun ~size -> match size with
    | 1 -> (module UInt8)
    | 2 -> (module UInt16)
    | 4 -> (module UInt32)
    | 8 -> (module UInt64)
    | _ -> assert false

external size_t_size : unit -> int = "ctypes_size_t_size"
external ushort_size : unit -> int = "ctypes_ushort_size"
external uint_size : unit -> int = "ctypes_uint_size"
external ulong_size : unit -> int = "ctypes_ulong_size"
external ulonglong_size : unit -> int = "ctypes_ulonglong_size"

module Size_t : S = (val pick ~size:(size_t_size ()))
module UChar : S = UInt8
module UShort : S = (val pick ~size:(ushort_size ()))
module UInt : S = (val pick ~size:(uint_size ()))
module ULong : S = (val pick ~size:(ulong_size ()))
module ULLong : S = (val pick ~size:(ulonglong_size ()))

type uchar = UChar.t
type uint8 = UInt8.t
type uint16 = UInt16.t
type uint32 = UInt32.t
type uint64 = UInt64.t
type size_t = Size_t.t
type ushort = UShort.t
type uint = UInt.t
type ulong = ULong.t
type ullong = ULLong.t

end
module Signed : sig 
#1 "signed.mli"
(*
 * Copyright (c) 2013 Jeremy Yallop.
 *
 * This file is distributed under the terms of the MIT License.
 * See the file LICENSE for details.
 *)

(** Types and operations for signed integers. *)

module type S = sig
  include Unsigned.S

  val neg : t -> t
  (** Unary negation. *)

  val abs : t -> t
  (** Return the absolute value of its argument. *)

  val minus_one : t
  (** The value -1 *)

  val min_int : t
  (** The smallest representable integer. *)

  val shift_right_logical : t -> int -> t
  (** {!shift_right_logical} [x] [y] shifts [x] to the right by [y] bits.  See
      {!Int32.shift_right_logical}. *)

  val of_nativeint : nativeint -> t
  (** Convert the given nativeint value to a signed integer. *)

  val to_nativeint : t -> nativeint
  (** Convert the given signed integer to a nativeint value. *)

  val of_int64 : int64 -> t
  (** Convert the given int64 value to a signed integer. *)

  val to_int64 : t -> int64
  (** Convert the given signed integer to an int64 value. *)
end
(** Signed integer operations *)

module Int : S with type t = int
(** Signed integer type and operations. *)

module Int32 : S with type t = int32
(** Signed 32-bit integer type and operations. *)

module Int64 : S with type t = int64
(** Signed 64-bit integer type and operations. *)

module SInt : S
(** C's signed integer type and operations. *)

module Long : S
(** The signed long integer type and operations. *)

module LLong : S
(** The signed long long integer type and operations. *)

type sint = SInt.t
(** C's signed integer type. *)

type long = Long.t
(** The signed long integer type. *)

type llong = LLong.t
(** The signed long long integer type. *)

end = struct
#1 "signed.ml"
(*
 * Copyright (c) 2013 Jeremy Yallop.
 *
 * This file is distributed under the terms of the MIT License.
 * See the file LICENSE for details.
 *)

module type S = sig
  include Unsigned.S

  val neg : t -> t
  val abs : t -> t
  val minus_one : t
  val min_int : t
  val shift_right_logical : t -> int -> t
  val of_nativeint : nativeint -> t
  val to_nativeint : t -> nativeint
  val of_int64 : int64 -> t
  val to_int64 : t -> int64
end

module type Basics = sig
  type t
  val add : t -> t -> t
  val sub : t -> t -> t
  val mul : t -> t -> t
  val div : t -> t -> t
  val rem : t -> t -> t
  val logand : t -> t -> t
  val logor : t -> t -> t
  val logxor : t -> t -> t
  val shift_left : t -> int -> t
  val shift_right : t -> int -> t
  val shift_right_logical : t -> int -> t
end

module MakeInfix(S : Basics) =
struct
  open S
  let (+) = add
  let (-) = sub
  let ( * ) = mul
  let (/) = div
  let (mod) = rem
  let (land) = logand
  let (lor) = logor
  let (lxor) = logxor
  let (lsl) = shift_left
  let (lsr) = shift_right_logical
  let (asr) = shift_right
end

module Int =
struct
  module Basics =
  struct
    type t = int
    let add = ( + )
    let sub = ( - )
    let mul = ( * )
    let div = ( / )
    let rem = ( mod )
    let max_int = Pervasives.max_int
    let min_int = Pervasives.min_int
    let logand = ( land )
    let logor = ( lor )
    let logxor = ( lxor )
    let shift_left = ( lsl )
    let shift_right = ( asr )
    let shift_right_logical = ( lsr )
    let of_int x = x
    let to_int x = x
    let of_string = int_of_string
    let to_string = string_of_int
    let zero = 0
    let one = 1
    let minus_one = -1
    let lognot = lnot
    let succ = Pervasives.succ
    let pred = Pervasives.pred
    let compare = Pervasives.compare
  end
  include Basics
  module Infix = MakeInfix(Basics)
  let to_int64 = Int64.of_int
  let of_int64 = Int64.to_int
  let to_nativeint = Nativeint.of_int
  let of_nativeint = Nativeint.to_int
  let abs = Pervasives.abs
  let neg x = -x
end

module Int32 = 
struct
  include Int32
  module Infix = MakeInfix(Int32)
  let of_nativeint = Nativeint.to_int32
  let to_nativeint = Nativeint.of_int32
  let of_int64 = Int64.to_int32
  let to_int64 = Int64.of_int32
end

module Int64 = 
struct
  include Int64
  module Infix = MakeInfix(Int64)
  let of_int64 x = x
  let to_int64 x = x
end

(* C guarantees that sizeof(t) == sizeof(unsigned t) *)
external int_size : unit -> int = "ctypes_uint_size"
external long_size : unit -> int = "ctypes_ulong_size"
external llong_size : unit -> int = "ctypes_ulonglong_size"

let pick : size:int -> (module S) =
  fun ~size -> match size with
    | 4 -> (module Int32)
    | 8 -> (module Int64)
    | _ -> assert false

module SInt = (val pick ~size:(int_size ()))
module Long = (val pick ~size:(long_size ()))
module LLong = (val pick ~size:(llong_size ()))

type sint = SInt.t
type long = Long.t
type llong = LLong.t

end
module Ctypes_primitive_types : sig 
#1 "ctypes_primitive_types.mli"
(*
 * Copyright (c) 2014 Jeremy Yallop.
 *
 * This file is distributed under the terms of the MIT License.
 * See the file LICENSE for details.
 *)

(* Representation of primitive C types.

   Internal representation, not for public use. *)

open Unsigned
open Signed

type _ prim =
 | Char : char prim
 | Schar : int prim
 | Uchar : uchar prim
 | Bool : bool prim
 | Short : int prim
 | Int : int prim
 | Long : long prim
 | Llong : llong prim
 | Ushort : ushort prim
 | Sint : sint prim
 | Uint : uint prim
 | Ulong : ulong prim
 | Ullong : ullong prim
 | Size_t : size_t prim
 | Int8_t : int prim
 | Int16_t : int prim
 | Int32_t : int32 prim
 | Int64_t : int64 prim
 | Uint8_t : uint8 prim
 | Uint16_t : uint16 prim
 | Uint32_t : uint32 prim
 | Uint64_t : uint64 prim
 | Camlint : int prim
 | Nativeint : nativeint prim
 | Float : float prim
 | Double : float prim
 | Complex32 : Complex.t prim
 | Complex64 : Complex.t prim

type _ ml_prim = 
  | ML_char :  char ml_prim
  | ML_complex :  Complex.t ml_prim
  | ML_float :  float ml_prim
  | ML_int :  int ml_prim
  | ML_int32 :  int32 ml_prim
  | ML_int64 :  int64 ml_prim
  | ML_llong :  llong ml_prim
  | ML_long :  long ml_prim
  | ML_sint : sint ml_prim
  | ML_nativeint :  nativeint ml_prim
  | ML_size_t :  size_t ml_prim
  | ML_uchar :  uchar ml_prim
  | ML_bool :  bool ml_prim
  | ML_uint :  uint ml_prim
  | ML_uint16 :  uint16 ml_prim
  | ML_uint32 :  uint32 ml_prim
  | ML_uint64 :  uint64 ml_prim
  | ML_uint8 :  uint8 ml_prim
  | ML_ullong :  ullong ml_prim
  | ML_ulong :  ulong ml_prim
  | ML_ushort :  ushort ml_prim

val ml_prim : 'a prim -> 'a ml_prim

end = struct
#1 "ctypes_primitive_types.ml"
(*
 * Copyright (c) 2014 Jeremy Yallop.
 *
 * This file is distributed under the terms of the MIT License.
 * See the file LICENSE for details.
 *)

open Unsigned
open Signed

type _ prim =
 | Char : char prim
 | Schar : int prim
 | Uchar : uchar prim
 | Bool : bool prim
 | Short : int prim
 | Int : int prim
 | Long : long prim
 | Llong : llong prim
 | Ushort : ushort prim
 | Sint : sint prim
 | Uint : uint prim
 | Ulong : ulong prim
 | Ullong : ullong prim
 | Size_t : size_t prim
 | Int8_t : int prim
 | Int16_t : int prim
 | Int32_t : int32 prim
 | Int64_t : int64 prim
 | Uint8_t : uint8 prim
 | Uint16_t : uint16 prim
 | Uint32_t : uint32 prim
 | Uint64_t : uint64 prim
 | Camlint : int prim
 | Nativeint : nativeint prim
 | Float : float prim
 | Double : float prim
 | Complex32 : Complex.t prim
 | Complex64 : Complex.t prim

type _ ml_prim = 
  | ML_char :  char ml_prim
  | ML_complex :  Complex.t ml_prim
  | ML_float :  float ml_prim
  | ML_int :  int ml_prim
  | ML_int32 :  int32 ml_prim
  | ML_int64 :  int64 ml_prim
  | ML_llong :  llong ml_prim
  | ML_long :  long ml_prim
  | ML_sint : sint ml_prim
  | ML_nativeint :  nativeint ml_prim
  | ML_size_t :  size_t ml_prim
  | ML_uchar :  uchar ml_prim
  | ML_bool :  bool ml_prim
  | ML_uint :  uint ml_prim
  | ML_uint16 :  uint16 ml_prim
  | ML_uint32 :  uint32 ml_prim
  | ML_uint64 :  uint64 ml_prim
  | ML_uint8 :  uint8 ml_prim
  | ML_ullong :  ullong ml_prim
  | ML_ulong :  ulong ml_prim
  | ML_ushort :  ushort ml_prim

let ml_prim : type a. a prim -> a ml_prim = function
  | Char -> ML_char
  | Schar -> ML_int
  | Uchar -> ML_uchar
  | Bool -> ML_bool
  | Short -> ML_int
  | Int -> ML_int
  | Long -> ML_long
  | Llong -> ML_llong
  | Ushort -> ML_ushort
  | Sint -> ML_sint
  | Uint -> ML_uint
  | Ulong -> ML_ulong
  | Ullong -> ML_ullong
  | Size_t -> ML_size_t
  | Int8_t -> ML_int
  | Int16_t -> ML_int
  | Int32_t -> ML_int32
  | Int64_t -> ML_int64
  | Uint8_t -> ML_uint8
  | Uint16_t -> ML_uint16
  | Uint32_t -> ML_uint32
  | Uint64_t -> ML_uint64
  | Camlint -> ML_int
  | Nativeint -> ML_nativeint
  | Float -> ML_float
  | Double -> ML_float
  | Complex32 -> ML_complex
  | Complex64 -> ML_complex

end
module Ctypes_ptr
= struct
#1 "ctypes_ptr.ml"
(*
 * Copyright (c) 2013 Jeremy Yallop.
 *
 * This file is distributed under the terms of the MIT License.
 * See the file LICENSE for details.
 *)

(* Boxed pointers to C memory locations . *)

module Raw :
sig
  include Signed.S
  val null : t
end =
struct
  include Nativeint

  module Infix =
  struct
    let (+) = add
    let (-) = sub
    let ( * ) = mul
    let (/) = div
    let (mod) = rem
    let (land) = logand
    let (lor) = logor
    let (lxor) = logxor
    let (lsl) = shift_left
    let (lsr) = shift_right_logical
  end

  let of_nativeint x = x
  let to_nativeint x = x
  let of_int64 = Int64.to_nativeint
  let to_int64  = Int64.of_nativeint

  let null = zero
end

type voidp = Raw.t

module Fat :
sig
  (** A fat pointer, which holds a reference to the reference type, the C memory
      location, and an OCaml object. *)
  type _ t

  (** [make ?managed ~reftyp raw] builds a fat pointer from the reference
      type [reftyp], the C memory location [raw], and (optionally) an OCaml
      value, [managed].  The [managed] argument may be used to manage the
      lifetime of the C object; a typical use it to attach a finaliser to
      [managed] which releases the memory associated with the C object whose
      address is stored in [raw_ptr]. *)
  val make : ?managed:_ -> reftyp:'typ -> voidp -> 'typ t

  val is_null : _ t -> bool

  val reftype : 'typ t -> 'typ

  val managed : _ t -> Obj.t option

  val coerce : _ t -> 'typ -> 'typ t

  (** Return the raw pointer address.  The function is unsafe in the sense
      that it dissociates the address from the value which manages the memory,
      which may trigger associated finalisers, invalidating the address. *)
  val unsafe_raw_addr : _ t -> voidp

  val add_bytes : 'typ t -> int -> 'typ t

  val compare : 'typ t -> 'typ t -> int

  val diff_bytes : 'typ t -> 'typ t -> int
end =
struct
  type 'typ t =
    { reftyp  : 'typ;
      raw     : voidp;
      managed : Obj.t option; }

  let make ?managed ~reftyp raw = match managed with
    | None   -> { reftyp; raw; managed = None }
    | Some v -> { reftyp; raw; managed = Some (Obj.repr v) }

  let is_null { raw } = Raw.(compare zero) raw = 0

  let reftype { reftyp } = reftyp

  let managed { managed } = managed

  let coerce p reftyp = { p with reftyp }

  let unsafe_raw_addr { raw } = raw

  let add_bytes p bytes = { p with raw = Raw.(add p.raw (of_int bytes)) }

  let compare l r = Raw.compare l.raw r.raw

  let diff_bytes l r = Raw.(to_int (sub r.raw l.raw))
end

end
module Ctypes_bigarray_stubs
= struct
#1 "ctypes_bigarray_stubs.ml"
(*
 * Copyright (c) 2013 Jeremy Yallop.
 *
 * This file is distributed under the terms of the MIT License.
 * See the file LICENSE for details.
 *)

type _ kind =
    Kind_float32 : float kind
  | Kind_float64 : float kind
  | Kind_int8_signed : int kind
  | Kind_int8_unsigned : int kind
  | Kind_int16_signed : int kind
  | Kind_int16_unsigned : int kind
  | Kind_int32 : int32 kind
  | Kind_int64 : int64 kind
  | Kind_int : int kind
  | Kind_nativeint : nativeint kind
  | Kind_complex32 : Complex.t kind
  | Kind_complex64 : Complex.t kind
  | Kind_char : char kind

external kind : ('a, 'b) Bigarray.kind -> 'a kind
  (* Bigarray.kind is simply an int whose values are consecutively numbered
     starting from zero, so we can directly transform its values to a variant
     with appropriately-ordered constructors.

     In OCaml <= 4.01.0, Bigarray.char and Bigarray.int8_unsigned are
     indistinguishable, so the 'kind' function will never return Kind_char.
     OCaml 4.02.0 gives the types distinct representations. *)
  = "%identity"

external address : 'b -> Ctypes_ptr.voidp
  = "ctypes_bigarray_address"

external view : 'a kind -> dims:int array -> _ Ctypes_ptr.Fat.t ->
  ('a, 'b, Bigarray.c_layout) Bigarray.Genarray.t
  = "ctypes_bigarray_view"

external view1 : 'a kind -> dims:int array -> _ Ctypes_ptr.Fat.t ->
  ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t
  = "ctypes_bigarray_view"

external view2 : 'a kind -> dims:int array -> _ Ctypes_ptr.Fat.t ->
  ('a, 'b, Bigarray.c_layout) Bigarray.Array2.t
  = "ctypes_bigarray_view"

external view3 : 'a kind -> dims:int array -> _ Ctypes_ptr.Fat.t ->
  ('a, 'b, Bigarray.c_layout) Bigarray.Array3.t
  = "ctypes_bigarray_view"

end
module Ctypes_memory_stubs
= struct
#1 "ctypes_memory_stubs.ml"
(*
 * Copyright (c) 2013 Jeremy Yallop.
 *
 * This file is distributed under the terms of the MIT License.
 * See the file LICENSE for details.
 *)

(* Stubs for reading and writing memory. *)

open Ctypes_ptr

(* A reference, managed by the garbage collector, to a region of memory in the
   C heap. *)
type managed_buffer

(* Allocate a region of stable, zeroed memory managed by a custom block. *)
external allocate : int -> int -> managed_buffer
  = "ctypes_allocate"

(* Obtain the address of the managed block. *)
external block_address : managed_buffer -> voidp
  = "ctypes_block_address"

(* Read a C value from a block of memory *)
external read : 'a Ctypes_primitive_types.prim -> _ Fat.t -> 'a
  = "ctypes_read"

(* Write a C value to a block of memory *)
external write : 'a Ctypes_primitive_types.prim -> 'a -> _ Fat.t -> unit
  = "ctypes_write"

module Pointer =
struct
  external read : _ Fat.t -> voidp
    = "ctypes_read_pointer"

  external write : _ Fat.t -> _ Fat.t -> unit
  = "ctypes_write_pointer"
end

(* Copy [size] bytes from [src] to [dst]. *)
external memcpy : dst:_ Fat.t -> src:_ Fat.t -> size:int -> unit
  = "ctypes_memcpy"

(* Read a fixed length OCaml string from memory *)
external string_of_array : _ Fat.t -> len:int -> string
  = "ctypes_string_of_array"

(* Do nothing, concealing from the optimizer that nothing is being done. *)
external use_value : 'a -> unit
  = "ctypes_use"

end
module Ctypes_path : sig 
#1 "ctypes_path.mli"
(*
 * Copyright (c) 2014 Jeremy Yallop.
 *
 * This file is distributed under the terms of the MIT License.
 * See the file LICENSE for details.
 *)

(* Value paths (long identifiers) *)

type path

val path_of_string : string -> path
val format_path : Format.formatter -> path -> unit

end = struct
#1 "ctypes_path.ml"
(*
 * Copyright (c) 2014 Jeremy Yallop.
 *
 * This file is distributed under the terms of the MIT License.
 * See the file LICENSE for details.
 *)

(* Paths (long identifiers) *)

type path = string list

let is_uident s =
  Str.(string_match (regexp "[A-Z][a-zA-Z0-9_]*") s 0);;

let is_ident s =
  Str.(string_match (regexp "[A-Za-z_][a-zA-Z0-9_]*") s 0);;

let rec is_valid_path = function
  | [] -> false
  | [l] -> is_ident l
  | u :: p -> is_uident u && is_valid_path p

let path_of_string s = 
  let p = Str.(split (regexp_string ".") s) in
  if is_valid_path p then p
  else invalid_arg "Ctypes_ident.path_of_string"

let format_path fmt p =
  Format.pp_print_string fmt (String.concat "." p)

end
module Ctypes_primitives
= struct
#1 "ctypes_primitives.ml"
(*
 * Copyright (c) 2016 whitequark
 *
 * This file is distributed under the terms of the MIT License.
 * See the file LICENSE for details.
 *)

open Ctypes_primitive_types
let sizeof : type a. a prim -> int = function
 | Char -> 1
 | Schar -> 1
 | Uchar -> 1
 | Bool -> 1
 | Short -> 2
 | Int -> 4
 | Long -> 8
 | Llong -> 8
 | Ushort -> 2
 | Sint -> 4
 | Uint -> 4
 | Ulong -> 8
 | Ullong -> 8
 | Size_t -> 8
 | Int8_t -> 1
 | Int16_t -> 2
 | Int32_t -> 4
 | Int64_t -> 8
 | Uint8_t -> 1
 | Uint16_t -> 2
 | Uint32_t -> 4
 | Uint64_t -> 8
 | Float -> 4
 | Double -> 8
 | Complex32 -> 8
 | Complex64 -> 16
 | Nativeint -> 8
 | Camlint -> 8
let alignment : type a. a prim -> int = function
 | Char -> 1
 | Schar -> 1
 | Uchar -> 1
 | Bool -> 1
 | Short -> 2
 | Int -> 4
 | Long -> 8
 | Llong -> 8
 | Ushort -> 2
 | Sint -> 4
 | Uint -> 4
 | Ulong -> 8
 | Ullong -> 8
 | Size_t -> 8
 | Int8_t -> 1
 | Int16_t -> 2
 | Int32_t -> 4
 | Int64_t -> 8
 | Uint8_t -> 1
 | Uint16_t -> 2
 | Uint32_t -> 4
 | Uint64_t -> 8
 | Float -> 4
 | Double -> 8
 | Complex32 -> 4
 | Complex64 -> 8
 | Nativeint -> 8
 | Camlint -> 8
let name : type a. a prim -> string = function
 | Char -> "char"
 | Schar -> "signed char"
 | Uchar -> "unsigned char"
 | Bool -> "_Bool"
 | Short -> "short"
 | Int -> "int"
 | Long -> "long"
 | Llong -> "long long"
 | Ushort -> "unsigned short"
 | Sint -> "int"
 | Uint -> "unsigned int"
 | Ulong -> "unsigned long"
 | Ullong -> "unsigned long long"
 | Size_t -> "size_t"
 | Int8_t -> "int8_t"
 | Int16_t -> "int16_t"
 | Int32_t -> "int32_t"
 | Int64_t -> "int64_t"
 | Uint8_t -> "uint8_t"
 | Uint16_t -> "uint16_t"
 | Uint32_t -> "uint32_t"
 | Uint64_t -> "uint64_t"
 | Float -> "float"
 | Double -> "double"
 | Complex32 -> "float _Complex"
 | Complex64 -> "double _Complex"
 | Nativeint -> "intnat"
 | Camlint -> "camlint"
let format_string : type a. a prim -> string option = function
 | Char -> Some "%d"
 | Schar -> Some "%d"
 | Uchar -> Some "%d"
 | Bool -> Some "%d"
 | Short -> Some "%hd"
 | Int -> Some "%d"
 | Long -> Some "%ld"
 | Llong -> Some "%lld"
 | Ushort -> Some "%hu"
 | Sint -> Some "%d"
 | Uint -> Some "%u"
 | Ulong -> Some "%lu"
 | Ullong -> Some "%llu"
 | Size_t -> Some "%zu"
 | Int8_t -> Some "%hhd"
 | Int16_t -> Some "%hd"
 | Int32_t -> Some "%d"
 | Int64_t -> Some "%lld"
 | Uint8_t -> Some "%hhu"
 | Uint16_t -> Some "%hu"
 | Uint32_t -> Some "%u"
 | Uint64_t -> Some "%llu"
 | Float -> Some "%.12g"
 | Double -> Some "%.12g"
 | Complex32 -> None
 | Complex64 -> None
 | Nativeint -> Some "%ld"
 | Camlint -> Some "%ld"
let pointer_size = 8
let pointer_alignment = 8

end
module Ctypes_bigarray : sig 
#1 "ctypes_bigarray.mli"
(*
 * Copyright (c) 2013 Jeremy Yallop.
 *
 * This file is distributed under the terms of the MIT License.
 * See the file LICENSE for details.
 *)

(** {2 Types *)

type ('a, 'b) t
(** The type of bigarray values of particular sizes.  A value of type
    [(a, b) t] can be used to read and write values of type [b].  *)

(** {3 Type constructors *)

val bigarray : int array -> ('a, 'b) Bigarray.kind ->
  ('a, ('a, 'b, Bigarray.c_layout) Bigarray.Genarray.t) t
(** Create a {!t} value for the {!Bigarray.Genarray.t} type. *)

val bigarray1 : int -> ('a, 'b) Bigarray.kind ->
  ('a, ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t) t
(** Create a {!t} value for the {!Bigarray.Array1.t} type. *)

val bigarray2 : int -> int -> ('a, 'b) Bigarray.kind ->
  ('a, ('a, 'b, Bigarray.c_layout) Bigarray.Array2.t) t
(** Create a {!t} value for the {!Bigarray.Array2.t} type. *)

val bigarray3 : int -> int -> int -> ('a, 'b) Bigarray.kind ->
  ('a, ('a, 'b, Bigarray.c_layout) Bigarray.Array3.t) t
(** Create a {!t} value for the {!Bigarray.Array3.t} type. *)

val prim_of_kind : ('a, _) Bigarray.kind -> 'a Ctypes_primitive_types.prim
(** Create a {!Ctypes_ptr.Types.ctype} for a {!Bigarray.kind}. *)

(** {3 Type eliminators *)

val sizeof : (_, _) t -> int
(** Compute the size of a bigarray type. *)

val alignment : (_, _) t -> int
(** Compute the alignment of a bigarray type. *)

val element_type : ('a, _) t -> 'a Ctypes_primitive_types.prim
(** Compute the element type of a bigarray type. *)

val dimensions : (_, _) t -> int array
(** Compute the dimensions of a bigarray type. *)

val type_expression : ('a, 'b) t -> ([> `Appl of Ctypes_path.path * 'c list
                                     |  `Ident of Ctypes_path.path ] as 'c)
(** Compute a type expression that denotes a bigarray type. *)

(** {2 Values *)

val unsafe_address : 'a -> Ctypes_ptr.voidp
(** Return the address of a bigarray value.  This function is unsafe because
    it dissociates the raw address of the C array from the OCaml object that
    manages the lifetime of the array.  If the caller does not hold a
    reference to the OCaml object then the array might be freed, invalidating
    the address. *)

val view : (_, 'a) t -> _ Ctypes_ptr.Fat.t -> 'a
(** [view b ptr] creates a bigarray view onto existing memory.

    If [ptr] references an OCaml object then [view] will ensure that
    that object is not collected before the bigarray returned by
    [view]. *)

end = struct
#1 "ctypes_bigarray.ml"
(*
 * Copyright (c) 2013 Jeremy Yallop.
 *
 * This file is distributed under the terms of the MIT License.
 * See the file LICENSE for details.
 *)

open Ctypes_bigarray_stubs

let prim_of_kind : type a. a kind -> a Ctypes_primitive_types.prim
  = let open Ctypes_primitive_types in function
    Kind_float32 -> Float
  | Kind_float64 -> Double
  | Kind_int8_signed -> Int8_t
  | Kind_int8_unsigned -> Int8_t
  | Kind_int16_signed -> Int16_t
  | Kind_int16_unsigned -> Int16_t
  | Kind_int32 -> Int32_t
  | Kind_int64 -> Int64_t
  | Kind_int -> Camlint
  | Kind_nativeint -> Nativeint
  | Kind_complex32 -> Complex32
  | Kind_complex64 -> Complex64
  | Kind_char -> Char

let bigarray_kind_sizeof k = Ctypes_primitives.sizeof (prim_of_kind k)

let bigarray_kind_alignment k = Ctypes_primitives.alignment (prim_of_kind k)

type (_, _) dims = 
| DimsGen : int array -> ('a, ('a, _, Bigarray.c_layout) Bigarray.Genarray.t) dims
| Dims1 : int -> ('a, ('a, _, Bigarray.c_layout) Bigarray.Array1.t) dims
| Dims2 : int * int -> ('a, ('a, _, Bigarray.c_layout) Bigarray.Array2.t) dims
| Dims3 : int * int * int -> ('a, ('a, _, Bigarray.c_layout) Bigarray.Array3.t) dims

type ('a, 'b) t = ('a, 'b) dims * 'a kind

let elements : type a b. (b, a) dims -> int = function
  | DimsGen ds -> Array.fold_left ( * ) 1 ds
  | Dims1 d -> d
  | Dims2 (d1, d2) -> d1 * d2
  | Dims3 (d1, d2, d3) -> d1 * d2 * d3

let element_type (_, k) = prim_of_kind k

let dimensions : type a b. (b, a) t -> int array = function
| DimsGen dims, _ -> dims
| Dims1 x, _ -> [| x |]
| Dims2 (x, y), _ -> [| x; y |]
| Dims3 (x, y, z), _ -> [| x; y; z |]

let sizeof (d, k) = elements d * bigarray_kind_sizeof k

let alignment (d, k) = bigarray_kind_alignment k

let bigarray ds k = (DimsGen ds, kind k)
let bigarray1 d k = (Dims1 d, kind k)
let bigarray2 d1 d2 k = (Dims2 (d1, d2), kind k)
let bigarray3 d1 d2 d3 k = (Dims3 (d1, d2, d3), kind k)

let path_of_string = Ctypes_path.path_of_string
let type_name : type a b. (b, a) dims -> Ctypes_path.path = function
  | DimsGen _ -> path_of_string "Bigarray.Genarray.t"
  | Dims1 _ -> path_of_string "Bigarray.Array1.t"
  | Dims2 _ -> path_of_string "Bigarray.Array2.t"
  | Dims3 _ -> path_of_string "Bigarray.Array3.t"

let kind_type_names : type a. a kind -> _ = function
  | Kind_float32 ->
    (`Ident (path_of_string "float"),
     `Ident (path_of_string "Bigarray.float32_elt"))
  | Kind_float64 ->
    (`Ident (path_of_string "float"),
     `Ident (path_of_string "Bigarray.float64_elt"))
  | Kind_int8_signed ->
    (`Ident (path_of_string "int"),
     `Ident (path_of_string "Bigarray.int8_signed_elt"))
  | Kind_int8_unsigned ->
    (`Ident (path_of_string "int"),
     `Ident (path_of_string "Bigarray.int8_unsigned_elt"))
  | Kind_int16_signed ->
    (`Ident (path_of_string "int"),
     `Ident (path_of_string "Bigarray.int16_signed_elt"))
  | Kind_int16_unsigned ->
    (`Ident (path_of_string "int"),
     `Ident (path_of_string "Bigarray.int16_unsigned_elt"))
  | Kind_int32 ->
    (`Ident (path_of_string "int32"),
     `Ident (path_of_string "Bigarray.int32_elt"))
  | Kind_int64 ->
    (`Ident (path_of_string "int64"),
     `Ident (path_of_string "Bigarray.int64_elt"))
  | Kind_int ->
    (`Ident (path_of_string "int"),
     `Ident (path_of_string "Bigarray.int_elt"))
  | Kind_nativeint ->
    (`Ident (path_of_string "nativeint"),
     `Ident (path_of_string "Bigarray.nativeint_elt"))
  | Kind_complex32 ->
    (`Ident (path_of_string "Complex.t"),
     `Ident (path_of_string "Bigarray.complex32_elt"))
  | Kind_complex64 ->
    (`Ident (path_of_string "Complex.t"),
     `Ident (path_of_string "Bigarray.complex64_elt"))
  | Kind_char ->
    (`Ident (path_of_string "char"),
     `Ident (path_of_string "Bigarray.int8_unsigned_elt"))

let type_expression : type a b. (a, b) t -> _ =
  fun (t, ck) ->
  begin
    let a, b = kind_type_names ck in
    let layout = `Ident (path_of_string "Bigarray.c_layout") in
    (`Appl (type_name t, [a; b; layout])
        : [> `Ident of Ctypes_path.path
          | `Appl of Ctypes_path.path * 'a list ] as 'a)
  end

let prim_of_kind k = prim_of_kind (kind k)

let unsafe_address b = Ctypes_bigarray_stubs.address b

let view : type a b. (a, b) t -> _ Ctypes_ptr.Fat.t -> b =
  let open Ctypes_bigarray_stubs in
  fun (dims, kind) ptr -> let ba : b = match dims with
  | DimsGen ds -> view kind ~dims:ds ptr
  | Dims1 d -> view1 kind ~dims:[| d |] ptr
  | Dims2 (d1, d2) -> view2 kind ~dims:[| d1; d2 |] ptr
  | Dims3 (d1, d2, d3) -> view3 kind ~dims:[| d1; d2; d3 |] ptr in
  match Ctypes_ptr.Fat.managed ptr with
  | None -> ba
  | Some src -> Gc.finalise (fun _ -> Ctypes_memory_stubs.use_value src) ba; ba

end
module Ctypes_static : sig 
#1 "ctypes_static.mli"
(*
 * Copyright (c) 2013 Jeremy Yallop.
 *
 * This file is distributed under the terms of the MIT License.
 * See the file LICENSE for details.
 *)

(* C type construction.  Internal representation, not for public use. *)

type abstract_type = {
  aname : string;
  asize : int;
  aalignment : int;
}

type _ ocaml_type =
  String     : string ocaml_type
| Bytes      : Bytes.t ocaml_type
| FloatArray : float array ocaml_type

type incomplete_size = { mutable isize: int }

type structured_spec = { size: int; align: int; }

type 'a structspec =
    Incomplete of incomplete_size
  | Complete of structured_spec

type _ typ =
    Void            :                       unit typ
  | Primitive       : 'a Ctypes_primitive_types.prim -> 'a typ
  | Pointer         : 'a typ             -> 'a ptr typ
  | Funptr          : 'a fn              -> 'a static_funptr typ
  | Struct          : 'a structure_type  -> 'a structure typ
  | Union           : 'a union_type      -> 'a union typ
  | Abstract        : abstract_type      -> 'a abstract typ
  | View            : ('a, 'b) view      -> 'a typ
  | Array           : 'a typ * int       -> 'a carray typ
  | Bigarray        : (_, 'a) Ctypes_bigarray.t
                                         -> 'a typ
  | OCaml           : 'a ocaml_type      -> 'a ocaml typ
and 'a carray = { astart : 'a ptr; alength : int }
and ('a, 'kind) structured = { structured : ('a, 'kind) structured ptr }
and 'a union = ('a, [`Union]) structured
and 'a structure = ('a, [`Struct]) structured
and 'a abstract = ('a, [`Abstract]) structured
and (_, _) pointer =
  CPointer : 'a typ Ctypes_ptr.Fat.t -> ('a, [`C]) pointer
| OCamlRef : int * 'a * 'a ocaml_type -> ('a, [`OCaml]) pointer
and 'a ptr = ('a, [`C]) pointer
and 'a ocaml = ('a, [`OCaml]) pointer
and 'a static_funptr = Static_funptr of 'a fn Ctypes_ptr.Fat.t
and ('a, 'b) view = {
  read : 'b -> 'a;
  write : 'a -> 'b;
  format_typ: ((Format.formatter -> unit) -> Format.formatter -> unit) option;
  format: (Format.formatter -> 'a -> unit) option;
  ty: 'b typ;
}
and ('a, 's) field = {
  ftype: 'a typ;
  foffset: int;
  fname: string;
}
and 'a structure_type = {
  tag: string;
  mutable spec: 'a structspec;
  mutable fields : 'a structure boxed_field list;
}
and 'a union_type = {
  utag: string;
  mutable uspec: structured_spec option;
  mutable ufields : 'a union boxed_field list;
}
and 's boxed_field = BoxedField : ('a, 's) field -> 's boxed_field
and _ fn =
  | Returns  : 'a typ   -> 'a fn
  | Function : 'a typ * 'b fn  -> ('a -> 'b) fn

type _ bigarray_class =
  Genarray :
  < element: 'a;
    dims: int array;
    ba_repr: 'b;
    bigarray: ('a, 'b, Bigarray.c_layout) Bigarray.Genarray.t;
    carray: 'a carray > bigarray_class
| Array1 :
  < element: 'a;
    dims: int;
    ba_repr: 'b;
    bigarray: ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t;
    carray: 'a carray > bigarray_class
| Array2 :
  < element: 'a;
    dims: int * int;
    ba_repr: 'b;
    bigarray: ('a, 'b, Bigarray.c_layout) Bigarray.Array2.t;
    carray: 'a carray carray > bigarray_class
| Array3 :
  < element: 'a;
    dims: int * int * int;
    ba_repr: 'b;
    bigarray: ('a, 'b, Bigarray.c_layout) Bigarray.Array3.t;
    carray: 'a carray carray carray > bigarray_class

type boxed_typ = BoxedType : 'a typ -> boxed_typ

val sizeof : 'a typ -> int
val alignment : 'a typ -> int
val passable : 'a typ -> bool
val ocaml_value : 'a typ -> bool
val has_ocaml_argument : 'a fn -> bool

val void : unit typ
val char : char typ
val schar : int typ
val float : float typ
val double : float typ
val complex32 : Complex.t typ
val complex64 : Complex.t typ
val short : int typ
val int : int typ
val sint : Signed.sint typ
val long : Signed.long typ
val llong : Signed.llong typ
val nativeint : nativeint typ
val int8_t : int typ
val int16_t : int typ
val int32_t : Signed.Int32.t typ
val int64_t : Signed.Int64.t typ
val camlint : int typ
val uchar : Unsigned.uchar typ
val bool : bool typ
val uint8_t : Unsigned.UInt8.t typ
val uint16_t : Unsigned.UInt16.t typ
val uint32_t : Unsigned.UInt32.t typ
val uint64_t : Unsigned.UInt64.t typ
val size_t : Unsigned.size_t typ
val ushort : Unsigned.ushort typ
val uint : Unsigned.uint typ
val ulong : Unsigned.ulong typ
val ullong : Unsigned.ullong typ
val array : int -> 'a typ -> 'a carray typ
val ocaml_string : string ocaml typ
val ocaml_bytes : Bytes.t ocaml typ
val ocaml_float_array : float array ocaml typ
val ptr : 'a typ -> 'a ptr typ
val ( @-> ) : 'a typ -> 'b fn -> ('a -> 'b) fn
val abstract : name:string -> size:int -> alignment:int -> 'a abstract typ
val view : ?format_typ:((Format.formatter -> unit) ->
                        Format.formatter -> unit) ->
           ?format: (Format.formatter -> 'b -> unit) ->
           read:('a -> 'b) -> write:('b -> 'a) -> 'a typ -> 'b typ
val typedef: 'a typ -> string -> 'a typ
val bigarray : < ba_repr : 'c;
                 bigarray : 'd;
                 carray : 'e;
                 dims : 'b;
                 element : 'a > bigarray_class ->
               'b -> ('a, 'c) Bigarray.kind -> 'd typ
val returning : 'a typ -> 'a fn
val static_funptr : 'a fn -> 'a static_funptr typ
val structure : string -> 'a structure typ
val union : string -> 'a union typ
val offsetof : ('a, 'b) field -> int
val field_type : ('a, 'b) field -> 'a typ
val field_name : ('a, 'b) field -> string

exception IncompleteType
exception ModifyingSealedType of string
exception Unsupported of string

val unsupported : ('a, unit, string, _) format4 -> 'a

(* This corresponds to the enum in ctypes_primitives.h *)
type arithmetic =
    Int8
  | Int16
  | Int32
  | Int64
  | Uint8
  | Uint16
  | Uint32
  | Uint64
  | Float
  | Double

end = struct
#1 "ctypes_static.ml"
(*
 * Copyright (c) 2013 Jeremy Yallop.
 *
 * This file is distributed under the terms of the MIT License.
 * See the file LICENSE for details.
 *)

(* C type construction *)

exception IncompleteType
exception ModifyingSealedType of string
exception Unsupported of string

let unsupported fmt = Printf.ksprintf (fun s -> raise (Unsupported s)) fmt

type incomplete_size = { mutable isize: int }

type structured_spec = { size: int; align: int; }

type 'a structspec =
    Incomplete of incomplete_size
  | Complete of structured_spec

type abstract_type = {
  aname : string;
  asize : int;
  aalignment : int;
}

type _ ocaml_type =
  String     : string ocaml_type
| Bytes      : Bytes.t ocaml_type
| FloatArray : float array ocaml_type

type _ typ =
    Void            :                       unit typ
  | Primitive       : 'a Ctypes_primitive_types.prim -> 'a typ
  | Pointer         : 'a typ             -> 'a ptr typ
  | Funptr          : 'a fn              -> 'a static_funptr typ
  | Struct          : 'a structure_type  -> 'a structure typ
  | Union           : 'a union_type      -> 'a union typ
  | Abstract        : abstract_type      -> 'a abstract typ
  | View            : ('a, 'b) view      -> 'a typ
  | Array           : 'a typ * int       -> 'a carray typ
  | Bigarray        : (_, 'a) Ctypes_bigarray.t
                                         -> 'a typ
  | OCaml           : 'a ocaml_type      -> 'a ocaml typ
and 'a carray = { astart : 'a ptr; alength : int }
and ('a, 'kind) structured = { structured : ('a, 'kind) structured ptr }
and 'a union = ('a, [`Union]) structured
and 'a structure = ('a, [`Struct]) structured
and 'a abstract = ('a, [`Abstract]) structured
and (_, _) pointer =
  CPointer : 'a typ Ctypes_ptr.Fat.t -> ('a, [`C]) pointer
| OCamlRef : int * 'a * 'a ocaml_type -> ('a, [`OCaml]) pointer
and 'a ptr = ('a, [`C]) pointer
and 'a ocaml = ('a, [`OCaml]) pointer
and 'a static_funptr = Static_funptr of 'a fn Ctypes_ptr.Fat.t
and ('a, 'b) view = {
  read : 'b -> 'a;
  write : 'a -> 'b;
  format_typ: ((Format.formatter -> unit) -> Format.formatter -> unit) option;
  format: (Format.formatter -> 'a -> unit) option;
  ty: 'b typ;
}
and ('a, 's) field = {
  ftype: 'a typ;
  foffset: int;
  fname: string;
}
and 'a structure_type = {
  tag: string;
  mutable spec: 'a structspec;
  (* fields are in reverse order iff the struct type is incomplete *)
  mutable fields : 'a structure boxed_field list;
}
and 'a union_type = {
  utag: string;
  mutable uspec: structured_spec option;
  (* fields are in reverse order iff the union type is incomplete *)
  mutable ufields : 'a union boxed_field list;
}
and 's boxed_field = BoxedField : ('a, 's) field -> 's boxed_field
and _ fn =
  | Returns  : 'a typ   -> 'a fn
  | Function : 'a typ * 'b fn  -> ('a -> 'b) fn

type _ bigarray_class =
  Genarray :
  < element: 'a;
    dims: int array;
    ba_repr: 'b;
    bigarray: ('a, 'b, Bigarray.c_layout) Bigarray.Genarray.t;
    carray: 'a carray > bigarray_class
| Array1 :
  < element: 'a;
    dims: int;
    ba_repr: 'b;
    bigarray: ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t;
    carray: 'a carray > bigarray_class
| Array2 :
  < element: 'a;
    dims: int * int;
    ba_repr: 'b;
    bigarray: ('a, 'b, Bigarray.c_layout) Bigarray.Array2.t;
    carray: 'a carray carray > bigarray_class
| Array3 :
  < element: 'a;
    dims: int * int * int;
    ba_repr: 'b;
    bigarray: ('a, 'b, Bigarray.c_layout) Bigarray.Array3.t;
    carray: 'a carray carray carray > bigarray_class

type boxed_typ = BoxedType : 'a typ -> boxed_typ

let rec sizeof : type a. a typ -> int = function
    Void                           -> raise IncompleteType
  | Primitive p                    -> Ctypes_primitives.sizeof p
  | Struct { spec = Incomplete _ } -> raise IncompleteType
  | Struct { spec = Complete
      { size } }                   -> size
  | Union { uspec = None }         -> raise IncompleteType
  | Union { uspec = Some { size } }
                                   -> size
  | Array (t, i)                   -> i * sizeof t
  | Bigarray ba                    -> Ctypes_bigarray.sizeof ba
  | Abstract { asize }             -> asize
  | Pointer _                      -> Ctypes_primitives.pointer_size
  | Funptr _                       -> Ctypes_primitives.pointer_size
  | OCaml _                        -> raise IncompleteType
  | View { ty }                    -> sizeof ty

let rec alignment : type a. a typ -> int = function
    Void                             -> raise IncompleteType
  | Primitive p                      -> Ctypes_primitives.alignment p
  | Struct { spec = Incomplete _ }   -> raise IncompleteType
  | Struct { spec = Complete
      { align } }                    -> align
  | Union { uspec = None }           -> raise IncompleteType
  | Union { uspec = Some { align } } -> align
  | Array (t, _)                     -> alignment t
  | Bigarray ba                      -> Ctypes_bigarray.alignment ba
  | Abstract { aalignment }          -> aalignment
  | Pointer _                        -> Ctypes_primitives.pointer_alignment
  | Funptr _                         -> Ctypes_primitives.pointer_alignment
  | OCaml _                          -> raise IncompleteType
  | View { ty }                      -> alignment ty

let rec passable : type a. a typ -> bool = function
    Void                           -> true
  | Primitive _                    -> true
  | Struct { spec = Incomplete _ } -> raise IncompleteType
  | Struct { spec = Complete _ }   -> true
  | Union  { uspec = None }        -> raise IncompleteType
  | Union  { uspec = Some _ }      -> true
  | Array _                        -> false
  | Bigarray _                     -> false
  | Pointer _                      -> true
  | Funptr _                       -> true
  | Abstract _                     -> false
  | OCaml _                        -> true
  | View { ty }                    -> passable ty

(* Whether a value resides in OCaml-managed memory.
   Values that reside in OCaml memory cannot be accessed
   when the runtime lock is not held. *)
let rec ocaml_value : type a. a typ -> bool = function
    Void        -> false
  | Primitive _ -> false
  | Struct _    -> false
  | Union _     -> false
  | Array _     -> false
  | Bigarray _  -> false
  | Pointer _   -> false
  | Funptr _    -> false
  | Abstract _  -> false
  | OCaml _     -> true
  | View { ty } -> ocaml_value ty

let rec has_ocaml_argument : type a. a fn -> bool = function
    Returns _ -> false
  | Function (t, _) when ocaml_value t -> true
  | Function (_, t) -> has_ocaml_argument t

let void = Void
let char = Primitive Ctypes_primitive_types.Char
let schar = Primitive Ctypes_primitive_types.Schar
let float = Primitive Ctypes_primitive_types.Float
let double = Primitive Ctypes_primitive_types.Double
let complex32 = Primitive Ctypes_primitive_types.Complex32
let complex64 = Primitive Ctypes_primitive_types.Complex64
let short = Primitive Ctypes_primitive_types.Short
let int = Primitive Ctypes_primitive_types.Int
let sint = Primitive Ctypes_primitive_types.Sint
let long = Primitive Ctypes_primitive_types.Long
let llong = Primitive Ctypes_primitive_types.Llong
let nativeint = Primitive Ctypes_primitive_types.Nativeint
let int8_t = Primitive Ctypes_primitive_types.Int8_t
let int16_t = Primitive Ctypes_primitive_types.Int16_t
let int32_t = Primitive Ctypes_primitive_types.Int32_t
let int64_t = Primitive Ctypes_primitive_types.Int64_t
let camlint = Primitive Ctypes_primitive_types.Camlint
let uchar = Primitive Ctypes_primitive_types.Uchar
let bool = Primitive Ctypes_primitive_types.Bool
let uint8_t = Primitive Ctypes_primitive_types.Uint8_t
let uint16_t = Primitive Ctypes_primitive_types.Uint16_t
let uint32_t = Primitive Ctypes_primitive_types.Uint32_t
let uint64_t = Primitive Ctypes_primitive_types.Uint64_t
let size_t = Primitive Ctypes_primitive_types.Size_t
let ushort = Primitive Ctypes_primitive_types.Ushort
let uint = Primitive Ctypes_primitive_types.Uint
let ulong = Primitive Ctypes_primitive_types.Ulong
let ullong = Primitive Ctypes_primitive_types.Ullong
let array i t = Array (t, i)
let ocaml_string = OCaml String
let ocaml_bytes = OCaml Bytes
let ocaml_float_array = OCaml FloatArray
let ptr t = Pointer t
let ( @->) f t =
  if not (passable f) then
    raise (Unsupported "Unsupported argument type")
  else
    Function (f, t)
let abstract ~name ~size ~alignment =
  Abstract { aname = name; asize = size; aalignment = alignment }
let view ?format_typ ?format ~read ~write ty =
  View { read; write; format_typ; format; ty }
let id v = v
let typedef old name =
  view ~format_typ:(fun k fmt -> Format.fprintf fmt "%s%t" name k)
    ~read:id ~write:id old
let bigarray : type a b c d e.
  < element: a;
    dims: b;
    ba_repr: c;
    bigarray: d;
    carray: e > bigarray_class ->
   b -> (a, c) Bigarray.kind -> d typ =
  fun spec dims kind -> match spec with
  | Genarray -> Bigarray (Ctypes_bigarray.bigarray dims kind)
  | Array1 -> Bigarray (Ctypes_bigarray.bigarray1 dims kind)
  | Array2 -> let d1, d2 = dims in
              Bigarray (Ctypes_bigarray.bigarray2 d1 d2 kind)
  | Array3 -> let d1, d2, d3 = dims in
              Bigarray (Ctypes_bigarray.bigarray3 d1 d2 d3 kind)
let returning v =
  if not (passable v) then
    raise (Unsupported "Unsupported return type")
  else
    Returns v
let static_funptr fn = Funptr fn

let structure tag =
  Struct { spec = Incomplete { isize = 0 }; tag; fields = [] }

let union utag = Union { utag; uspec = None; ufields = [] }

let offsetof { foffset } = foffset
let field_type { ftype } = ftype
let field_name { fname } = fname

(* This corresponds to the enum in ctypes_primitives.h *)
type arithmetic =
    Int8
  | Int16
  | Int32
  | Int64
  | Uint8
  | Uint16
  | Uint32
  | Uint64
  | Float
  | Double

end
module Ctypes_type_printing : sig 
#1 "ctypes_type_printing.mli"
(*
 * Copyright (c) 2014 Jeremy Yallop.
 *
 * This file is distributed under the terms of the MIT License.
 * See the file LICENSE for details.
 *)

open Ctypes_static

(* The format context affects the formatting of pointer, struct and union
   types.  There are three printing contexts: *)
type format_context = [
(* In the top-level context struct and union types are printed in full, with
   member lists.  Pointer types are unparenthesized; for example,
   pointer-to-void is printed as "void *", not as "void ( * )". *)
| `toplevel
(* In the array context, struct and union types are printed in abbreviated
   form, which consists of just a keyword and the tag name.  Pointer types are
   parenthesized; for example, pointer-to-array-of-int is printed as
   "int ( * )[]", not as "int *[]". *)
| `array
(* In the non-array context, struct and union types are printed in abbreviated
   form and pointer types are unparenthesized. *)
| `nonarray]

val format_name : ?name:string -> Format.formatter -> unit

val format_typ' : 'a Ctypes_static.typ -> (format_context -> Format.formatter -> unit) ->
  format_context -> Format.formatter -> unit

val format_typ : ?name:string -> Format.formatter -> 'a typ -> unit

val format_fn' : 'a fn -> (Format.formatter -> unit) -> Format.formatter -> unit

val format_fn : ?name:string -> Format.formatter -> 'a fn -> unit

val string_of_typ : ?name:string -> 'a typ -> string

val string_of_fn : ?name:string -> 'a fn -> string

end = struct
#1 "ctypes_type_printing.ml"
(*
 * Copyright (c) 2013 Jeremy Yallop.
 *
 * This file is distributed under the terms of the MIT License.
 * See the file LICENSE for details.
 *)

open Ctypes_static

(* See type_printing.mli for the documentation of [format context]. *)
type format_context = [ `toplevel | `array | `nonarray ]

let rec format_typ' : type a. a typ ->
  (format_context -> Format.formatter -> unit) ->
  (format_context -> Format.formatter -> unit) =
  let fprintf = Format.fprintf in
  fun t k context fmt -> match t with
    | Void ->
      fprintf fmt "void%t" (k `nonarray)
    | Primitive p ->
      let name = Ctypes_primitives.name p in
      fprintf fmt "%s%t" name (k `nonarray)
    | View { format_typ = Some format } ->
      format (k `nonarray) fmt
    | View { ty } ->
      format_typ' ty k context fmt
    | Abstract { aname } ->
      fprintf fmt "%s%t" aname (k `nonarray)
    | Struct { tag ; spec; fields } ->
      begin match spec, context with
        | Complete _, `toplevel ->
          begin
            fprintf fmt "struct %s {@;<1 2>@[" tag;
            format_fields fields fmt;
            fprintf fmt "@]@;}%t" (k `nonarray)
          end
        | _ -> fprintf fmt "struct %s%t" tag (k `nonarray)
      end
    | Union { utag; uspec; ufields } ->
      begin match uspec, context with
        | Some _, `toplevel ->
          begin
            fprintf fmt "union %s {@;<1 2>@[" utag;
            format_fields ufields fmt;
            fprintf fmt "@]@;}%t" (k `nonarray)
          end
        | _ -> fprintf fmt "union %s%t" utag (k `nonarray)
      end
    | Pointer ty ->
      format_typ' ty
        (fun context fmt ->
          match context with
            | `array -> fprintf fmt "(*%t)" (k `nonarray)
            | _      -> fprintf fmt "*%t" (k `nonarray))
        `nonarray fmt
    | Funptr fn ->
      format_fn' fn
        (fun fmt -> Format.fprintf fmt "(*%t)" (k `nonarray)) fmt
    | Array (ty, n) ->
      format_typ' ty (fun _ fmt -> fprintf fmt "%t[%d]" (k `array) n) `nonarray
        fmt
    | Bigarray ba ->
      let elem = Ctypes_bigarray.element_type ba
      and dims = Ctypes_bigarray.dimensions ba in
      let name = Ctypes_primitives.name elem in
      fprintf fmt "%s%t%t" name (k `array)
        (fun fmt -> (Array.iter (Format.fprintf fmt "[%d]") dims))
    | OCaml String -> format_typ' (ptr char) k context fmt
    | OCaml Bytes -> format_typ' (ptr char) k context fmt
    | OCaml FloatArray -> format_typ' (ptr double) k context fmt

and format_fields : type a. a boxed_field list -> Format.formatter -> unit =
  fun fields fmt ->
  let open Format in
      List.iteri
        (fun i (BoxedField {ftype=t; fname}) ->
          fprintf fmt "@[";
          format_typ' t (fun _ fmt -> fprintf fmt " %s" fname) `nonarray fmt;
          fprintf fmt "@];@;")
        fields
and format_parameter_list parameters k fmt =
  Format.fprintf fmt "%t(@[@[" k;
  if parameters = [] then Format.fprintf fmt "void" else
    List.iteri
      (fun i (BoxedType t) ->
        if i <> 0 then Format.fprintf fmt "@], @[";
        format_typ' t (fun _ _ -> ()) `nonarray fmt)
      parameters;
  Format.fprintf fmt "@]@])"
and format_fn' : 'a. 'a fn ->
  (Format.formatter -> unit) ->
  (Format.formatter -> unit) =
  let rec gather : type a. a fn -> boxed_typ list * boxed_typ =
    function
      | Returns ty -> [], BoxedType ty
      | Function (Void, fn) -> gather fn
      | Function (p, fn) -> let ps, r = gather fn in BoxedType p :: ps, r in
  fun fn k fmt ->
    let ps, BoxedType r = gather fn in
    format_typ' r (fun context fmt -> format_parameter_list ps k fmt)
      `nonarray fmt

let format_name ?name fmt =
  match name with
    | Some name -> Format.fprintf fmt " %s" name
    | None      -> ()

let format_typ : ?name:string -> Format.formatter -> 'a typ -> unit
  = fun ?name fmt typ ->
    Format.fprintf fmt "@[";
    format_typ' typ (fun context -> format_name ?name) `toplevel fmt;
    Format.fprintf fmt "@]"

let format_fn : ?name:string -> Format.formatter -> 'a fn -> unit
  = fun ?name fmt fn ->
    Format.fprintf fmt "@[";
    format_fn' fn (format_name ?name) fmt;
    Format.fprintf fmt "@]"

let string_of_typ ?name ty = Format.asprintf "%a" (format_typ ?name) ty
let string_of_fn ?name fn = Format.asprintf "%a" (format_fn ?name) fn

end
module Ctypes_coerce
= struct
#1 "ctypes_coerce.ml"
(*
 * Copyright (c) 2013 Jeremy Yallop.
 *
 * This file is distributed under the terms of the MIT License.
 * See the file LICENSE for details.
 *)

(* Coercions *)

open Ctypes_static

type uncoercible_info =
    Types : _ typ * _ typ -> uncoercible_info
  | Functions : _ fn * _ fn -> uncoercible_info

exception Uncoercible of uncoercible_info

let show_uncoercible = function
    Uncoercible (Types (l, r)) ->
    let pr ty = Ctypes_type_printing.string_of_typ ty in
    Some (Format.sprintf
            "Coercion failure: %s is not coercible to %s" (pr l) (pr r))
  | Uncoercible (Functions (l, r)) ->
    let pr ty = Ctypes_type_printing.string_of_fn ty in
    Some (Format.sprintf
            "Coercion failure: %s is not coercible to %s" (pr l) (pr r))
  | _ -> None

let () = Printexc.register_printer show_uncoercible

let uncoercible : 'a 'b 'c. 'a typ -> 'b typ -> 'c =
  fun l r -> raise (Uncoercible (Types (l, r)))

let uncoercible_functions : 'a 'b 'c. 'a fn -> 'b fn -> 'c =
  fun l r -> raise (Uncoercible (Functions (l, r)))

let id x = x

type (_, _) coercion =
  | Id : ('a, 'a) coercion
  | Coercion : ('a -> 'b) -> ('a, 'b) coercion

let ml_prim_coercion :
  type a b. a Ctypes_primitive_types.ml_prim -> b Ctypes_primitive_types.ml_prim ->
  (a, b) coercion option =
  let open Ctypes_primitive_types in
  fun l r -> match l, r with
  | ML_char, ML_char -> Some Id
  | ML_complex, ML_complex -> Some Id
  | ML_float, ML_float -> Some Id
  | ML_int, ML_int -> Some Id
  | ML_int32, ML_int32 -> Some Id
  | ML_int64, ML_int64 -> Some Id
  | ML_llong, ML_llong -> Some Id
  | ML_long, ML_long -> Some Id
  | ML_nativeint, ML_nativeint -> Some Id
  | ML_size_t, ML_size_t -> Some Id
  | ML_uchar, ML_uchar -> Some Id
  | ML_bool, ML_bool -> Some Id
  | ML_uint, ML_uint -> Some Id
  | ML_uint16, ML_uint16 -> Some Id
  | ML_uint32, ML_uint32 -> Some Id
  | ML_uint64, ML_uint64 -> Some Id
  | ML_uint8, ML_uint8 -> Some Id
  | ML_ullong, ML_ullong -> Some Id
  | ML_ulong, ML_ulong -> Some Id
  | ML_ushort, ML_ushort -> Some Id
  | l, r -> None

let rec coercion : type a b. a typ -> b typ -> (a, b) coercion =
  fun atyp btyp -> match atyp, btyp with
  | _, Void -> Coercion ignore
  | Primitive l, Primitive r ->
    (match Ctypes_primitive_types.(ml_prim_coercion (ml_prim l) (ml_prim r)) with
       Some c -> c
     | None -> uncoercible atyp btyp)
  | View av, b ->
    begin match coercion av.ty b with
    | Id -> Coercion av.write
    | Coercion coerce -> Coercion (fun v -> coerce (av.write v))
    end
  | a, View bv ->
    begin match coercion a bv.ty with
    | Id -> Coercion bv.read
    | Coercion coerce -> Coercion (fun v -> bv.read (coerce v))
    end
  | Pointer a, Pointer b ->
    begin
      try
        begin match coercion a b with
        | Id -> Id
        | Coercion _ ->
          Coercion (fun (CPointer p) -> CPointer (Ctypes_ptr.Fat.coerce p b))
        end
      with Uncoercible _ ->
        Coercion (fun (CPointer p) -> CPointer (Ctypes_ptr.Fat.coerce p b))
    end
  | Pointer a, Funptr b ->
    Coercion (fun (CPointer p) -> Static_funptr (Ctypes_ptr.Fat.coerce p b))
  | Funptr a, Pointer b ->
    Coercion (fun (Static_funptr p) -> CPointer (Ctypes_ptr.Fat.coerce p b))
  | Funptr a, Funptr b ->
    begin
      try
        begin match fn_coercion a b with
        | Id -> Id
        | Coercion _ ->
          Coercion (fun (Static_funptr p) -> Static_funptr (Ctypes_ptr.Fat.coerce p b))
        end
      with Uncoercible _ ->
        Coercion (fun (Static_funptr p) -> Static_funptr (Ctypes_ptr.Fat.coerce p b))
    end
  | l, r -> uncoercible l r

and fn_coercion : type a b. a fn -> b fn -> (a, b) coercion =
  fun afn bfn -> match afn, bfn with
  | Function (af, at), Function (bf, bt) ->
    begin match coercion bf af, fn_coercion at bt with
    | Id, Id -> Id
    | Id, Coercion h ->
      Coercion (fun g x -> h (g x))
    | Coercion f, Id ->
      Coercion (fun g x -> g (f x))
    | Coercion f, Coercion h ->
      Coercion (fun g x -> h (g (f x)))
    end
  | Returns at, Returns bt -> coercion at bt
  | l, r -> uncoercible_functions l r

let coerce : type a b. a typ -> b typ -> a -> b =
  fun atyp btyp -> match coercion atyp btyp with
  | Id -> id
  | Coercion c -> c

let coerce_fn : type a b. a fn -> b fn -> a -> b =
  fun afn bfn -> match fn_coercion afn bfn with
  | Id -> id
  | Coercion c -> c

end
module Ctypes_roots_stubs
= struct
#1 "ctypes_roots_stubs.ml"
(*
 * Copyright (c) 2015 Jeremy Yallop.
 *
 * This file is distributed under the terms of the MIT License.
 * See the file LICENSE for details.
 *)

external root : 'a -> Ctypes_ptr.voidp =
  "ctypes_caml_roots_create"

external set : Ctypes_ptr.voidp -> 'a -> unit =
  "ctypes_caml_roots_set"

external get : Ctypes_ptr.voidp -> 'a =
  "ctypes_caml_roots_get"

external release : Ctypes_ptr.voidp -> unit =
  "ctypes_caml_roots_release"

end
module Ctypes_memory
= struct
#1 "ctypes_memory.ml"
(*
 * Copyright (c) 2013 Jeremy Yallop.
 *
 * This file is distributed under the terms of the MIT License.
 * See the file LICENSE for details.
 *)

open Ctypes_static

module Stubs = Ctypes_memory_stubs
module Raw = Ctypes_ptr.Raw
module Fat = Ctypes_ptr.Fat

let castp reftype (CPointer p) = CPointer (Fat.coerce p reftype)

(* Describes how to read a value, e.g. from a return buffer *)
let rec build : type a b. a typ -> b typ Fat.t -> a
 = function
    | Void ->
      fun _ -> ()
    | Primitive p -> Stubs.read p
    | Struct { spec = Incomplete _ } ->
      raise IncompleteType
    | Struct { spec = Complete { size } } as reftyp ->
      (fun buf ->
        let managed = Stubs.allocate 1 size in
        let dst = Fat.make ~managed ~reftyp (Stubs.block_address managed) in
        let () = Stubs.memcpy ~size ~dst ~src:buf in
        { structured = CPointer dst})
    | Pointer reftyp ->
      (fun buf -> CPointer (Fat.make ~reftyp (Stubs.Pointer.read buf)))
    | Funptr fn ->
      (fun buf -> Static_funptr (Fat.make ~reftyp:fn (Stubs.Pointer.read buf)))
    | View { read; ty } ->
      let buildty = build ty in
      (fun buf -> read (buildty buf))
    | OCaml _ -> (fun buf -> assert false)
    (* The following cases should never happen; non-struct aggregate
       types are excluded during type construction. *)
    | Union _ -> assert false
    | Array _ -> assert false
    | Bigarray _ -> assert false
    | Abstract _ -> assert false

let rec write : type a b. a typ -> a -> b Fat.t -> unit
  = let write_aggregate size { structured = CPointer src } dst =
      Stubs.memcpy ~size ~dst ~src
    in
    function
    | Void -> (fun _ _ -> ())
    | Primitive p -> Stubs.write p
    | Pointer _ ->
      (fun (CPointer p) dst -> Stubs.Pointer.write p dst)
    | Funptr _ ->
      (fun (Static_funptr p) dst -> Stubs.Pointer.write p dst)
    | Struct { spec = Incomplete _ } -> raise IncompleteType
    | Struct { spec = Complete _ } as s -> write_aggregate (sizeof s)
    | Union { uspec = None } -> raise IncompleteType
    | Union { uspec = Some { size } } -> write_aggregate size
    | Abstract { asize } -> write_aggregate asize
    | Array _ as a ->
      let size = sizeof a in
      (fun { astart = CPointer src } dst ->
        Stubs.memcpy ~size ~dst ~src)
    | Bigarray b as t ->
      let size = sizeof t in
      (fun ba dst ->
        let src = Fat.make ~managed:ba ~reftyp:Void
          (Ctypes_bigarray.unsafe_address ba)
        in
        Stubs.memcpy ~size ~dst ~src)
    | View { write = w; ty } ->
      let writety = write ty in
      (fun v -> writety (w v))
    | OCaml _ -> raise IncompleteType

let null : unit ptr = CPointer (Fat.make ~reftyp:Void Raw.null)

let rec (!@) : type a. a ptr -> a
  = fun (CPointer cptr as ptr) ->
    match Fat.reftype cptr with
      | Void -> raise IncompleteType
      | Union { uspec = None } -> raise IncompleteType
      | Struct { spec = Incomplete _ } -> raise IncompleteType
      | View { read; ty } -> read (!@ (CPointer (Fat.coerce cptr ty)))
      (* If it's a reference type then we take a reference *)
      | Union _ -> { structured = ptr }
      | Struct _ -> { structured = ptr }
      | Array (elemtype, alength) ->
        { astart = CPointer (Fat.coerce cptr elemtype); alength }
      | Bigarray b -> Ctypes_bigarray.view b cptr
      | Abstract _ -> { structured = ptr }
      | OCaml _ -> raise IncompleteType
      (* If it's a value type then we cons a new value. *)
      | _ -> build (Fat.reftype cptr) cptr

let ptr_diff : type a b. (a, b) pointer -> (a, b) pointer -> int
  = fun l r ->
    match l, r with
    | CPointer lp, CPointer rp ->
      (* We assume the pointers are properly aligned, or at least that
         the difference is a multiple of sizeof reftype. *)
      Fat.diff_bytes lp rp / sizeof (Fat.reftype lp)
    | OCamlRef (lo, l, _), OCamlRef (ro, r, _) ->
      if l != r then invalid_arg "Ctypes.ptr_diff";
      ro - lo

let (+@) : type a b. (a, b) pointer -> int -> (a, b) pointer
  = fun p x ->
    match p with
    | CPointer p ->
      CPointer (Fat.add_bytes p (x * sizeof (Fat.reftype p)))
    | OCamlRef (offset, obj, ty) ->
      OCamlRef (offset + x, obj, ty)

let (-@) p x = p +@ (-x)

let (<-@) : type a. a ptr -> a -> unit
  = fun (CPointer p) ->
    fun v -> write (Fat.reftype p) v p

let from_voidp = castp
let to_voidp p = castp Void p

let allocate_n
  : type a. ?finalise:(a ptr -> unit) -> a typ -> count:int -> a ptr
  = fun ?finalise reftyp ~count ->
    let package p =
      CPointer (Fat.make ~managed:p ~reftyp (Stubs.block_address p))
    in
    let finalise = match finalise with
      | Some f -> Gc.finalise (fun p -> f (package p))
      | None -> ignore
    in
    let p = Stubs.allocate count (sizeof reftyp) in begin
      finalise p;
      package p
    end

let allocate : type a. ?finalise:(a ptr -> unit) -> a typ -> a -> a ptr
  = fun ?finalise reftype v ->
    let p = allocate_n ?finalise ~count:1 reftype in begin
      p <-@ v;
      p
    end

let ptr_compare (CPointer l) (CPointer r) = Fat.(compare l r)

let reference_type (CPointer p) = Fat.reftype p

let ptr_of_raw_address addr =
  CPointer (Fat.make ~reftyp:Void (Raw.of_nativeint addr))

let funptr_of_raw_address addr =
  Static_funptr (Fat.make ~reftyp:(void @-> returning void) (Raw.of_nativeint addr))

let raw_address_of_ptr (CPointer p) =
  (* This is unsafe by definition: if the object to which [p] refers
     is collected at this point then the returned address is invalid.
     If there is an OCaml object associated with [p] then it is vital
     that the caller retains a reference to it. *)
  Raw.to_nativeint (Fat.unsafe_raw_addr p)

module CArray =
struct
  type 'a t = 'a carray

  let check_bound { alength } i =
    if i >= alength then
      invalid_arg "index out of bounds"

  let unsafe_get { astart } n = !@(astart +@ n)
  let unsafe_set { astart } n v = (astart +@ n) <-@ v

  let get arr n =
    check_bound arr n;
    unsafe_get arr n

  let set arr n v =
    check_bound arr n;
    unsafe_set arr n v

  let start { astart } = astart
  let length { alength } = alength
  let from_ptr astart alength = { astart; alength }

  let fill ({ alength } as arr) v =
    for i = 0 to alength - 1 do unsafe_set arr i v done

  let make : type a. ?finalise:(a t -> unit) -> a typ -> ?initial:a -> int -> a t
    = fun ?finalise reftype ?initial count ->
      let finalise = match finalise with
        | Some f -> Some (fun astart -> f { astart; alength = count } )
        | None -> None
      in
      let arr = { astart = allocate_n ?finalise ~count reftype;
                  alength = count } in
      match initial with
        | None -> arr
        | Some v -> fill arr v; arr

  let element_type { astart } = reference_type astart

  let of_list typ list =
    let arr = make typ (List.length list) in
    List.iteri (set arr) list;
    arr

  let to_list a =
    let l = ref [] in
    for i = length a - 1 downto 0 do
      l := get a i :: !l
    done;
    !l
end

let make ?finalise s =
  let finalise = match finalise with
    | Some f -> Some (fun structured -> f { structured })
    | None -> None in
  { structured = allocate_n ?finalise s ~count:1 }
let (|->) (CPointer p) { ftype; foffset } =
  CPointer (Fat.(add_bytes (Fat.coerce p ftype) foffset))

let (@.) { structured = p } f = p |-> f
let setf s field v = (s @. field) <-@ v
let getf s field = !@(s @. field)

let addr { structured } = structured

open Bigarray

let _bigarray_start kind ba =
  let raw_address = Ctypes_bigarray.unsafe_address ba in
  let reftyp = Primitive (Ctypes_bigarray.prim_of_kind kind) in
  CPointer (Fat.make ~managed:ba ~reftyp raw_address)

let bigarray_kind : type a b c d f.
  < element: a;
    ba_repr: f;
    bigarray: b;
    carray: c;
    dims: d > bigarray_class -> b -> (a, f) Bigarray.kind =
  function
  | Genarray -> Genarray.kind
  | Array1 -> Array1.kind
  | Array2 -> Array2.kind
  | Array3 -> Array3.kind

let bigarray_start spec ba = _bigarray_start (bigarray_kind spec ba) ba

let array_of_bigarray : type a b c d e.
  < element: a;
    ba_repr: e;
    bigarray: b;
    carray: c;
    dims: d > bigarray_class -> b -> c
  = fun spec ba ->
    let CPointer p as element_ptr =
      bigarray_start spec ba in
    match spec with
  | Genarray ->
    let ds = Genarray.dims ba in
    CArray.from_ptr element_ptr (Array.fold_left ( * ) 1 ds)
  | Array1 ->
    let d = Array1.dim ba in
    CArray.from_ptr element_ptr d
  | Array2 ->
    let d1 = Array2.dim1 ba and d2 = Array2.dim2 ba in
    CArray.from_ptr (castp (array d2 (Fat.reftype p)) element_ptr) d1
  | Array3 ->
    let d1 = Array3.dim1 ba and d2 = Array3.dim2 ba and d3 = Array3.dim3 ba in
    CArray.from_ptr (castp (array d2 (array d3 (Fat.reftype p))) element_ptr) d1

let bigarray_elements : type a b c d f.
   < element: a;
     ba_repr: f;
     bigarray: b;
     carray: c;
     dims: d > bigarray_class -> d -> int
  = fun spec dims -> match spec, dims with
   | Genarray, ds -> Array.fold_left ( * ) 1 ds
   | Array1, d -> d
   | Array2, (d1, d2) -> d1 * d2
   | Array3, (d1, d2, d3) -> d1 * d2 * d3

let bigarray_of_ptr spec dims kind ptr =
  !@ (castp (bigarray spec dims kind) ptr)

let array_dims : type a b c d f.
   < element: a;
     ba_repr: f;
     bigarray: b;
     carray: c carray;
     dims: d > bigarray_class -> c carray -> d =
   let unsupported () = raise (Unsupported "taking dimensions of non-array type") in
   fun spec a -> match spec with
   | Genarray -> [| a.alength |]
   | Array1 -> a.alength
   | Array2 ->
     begin match a.astart with
     | CPointer p ->
       begin match Fat.reftype p with
       | Array (_, n) -> (a.alength, n)
       | _ -> unsupported ()
       end
    end
   | Array3 ->
     begin match a.astart with
     | CPointer p ->
       begin match Fat.reftype p with
       |  Array (Array (_, m), n) -> (a.alength, n, m)
       | _ -> unsupported ()
       end
     end

let bigarray_of_array spec kind a =
  let dims = array_dims spec a in
  !@ (castp (bigarray spec dims kind) (CArray.start a))

let genarray = Genarray
let array1 = Array1
let array2 = Array2
let array3 = Array3
let typ_of_bigarray_kind k = Primitive (Ctypes_bigarray.prim_of_kind k)

let string_from_ptr (CPointer p) ~length:len =
  if len < 0 then invalid_arg "Ctypes.string_from_ptr"
  else Stubs.string_of_array p ~len

let ocaml_string_start str =
  OCamlRef (0, str, String)

let ocaml_bytes_start str =
  OCamlRef (0, str, Bytes)

let ocaml_float_array_start arr =
  OCamlRef (0, arr, FloatArray)

module Root =
struct
  module Stubs = Ctypes_roots_stubs

  (* Roots are not managed values so it's safe to call unsafe_raw_addr. *)
  let raw_addr : unit ptr -> Raw.t =
    fun (CPointer p) -> Fat.unsafe_raw_addr p

  let create : 'a. 'a -> unit ptr =
    fun v -> CPointer (Fat.make ~reftyp:void (Stubs.root v))

  let get : 'a. unit ptr -> 'a =
    fun p -> Stubs.get (raw_addr p)

  let set : 'a. unit ptr -> 'a -> unit =
    fun p v -> Stubs.set (raw_addr p) v
  
  let release : 'a. unit ptr -> unit =
    fun p -> Stubs.release (raw_addr p)
end

let is_null (CPointer p) = Fat.is_null p

end
module Ctypes_std_view_stubs
= struct
#1 "ctypes_std_view_stubs.ml"
(*
 * Copyright (c) 2013 Jeremy Yallop.
 *
 * This file is distributed under the terms of the MIT License.
 * See the file LICENSE for details.
 *)

(* Stubs for standard views. *)

(* Convert a C string to an OCaml string *)
external string_of_cstring : char Ctypes_static.typ Ctypes_ptr.Fat.t -> string
  = "ctypes_string_of_cstring"

(* Convert an OCaml string to a C string *)
external cstring_of_string : string -> Ctypes_memory_stubs.managed_buffer
  = "ctypes_cstring_of_string"

(* Size information for uintptr_t *)
external uintptr_t_size : unit -> int = "ctypes_uintptr_t_size"

(* Size information for uintptr_t *)
external intptr_t_size : unit -> int = "ctypes_intptr_t_size"

(* Size information for ptrdiff_t *)
external ptrdiff_t_size : unit -> int = "ctypes_ptrdiff_t_size"

end
module Ctypes_std_views
= struct
#1 "ctypes_std_views.ml"
(*
 * Copyright (c) 2013 Jeremy Yallop.
 *
 * This file is distributed under the terms of the MIT License.
 * See the file LICENSE for details.
 *)

let string_of_char_ptr (Ctypes_static.CPointer p) =
  Ctypes_std_view_stubs.string_of_cstring p

let char_ptr_of_string s =
  let managed = Ctypes_std_view_stubs.cstring_of_string s in
  Ctypes_static.CPointer (Ctypes_ptr.Fat.make ~managed ~reftyp:Ctypes_static.char
                     (Ctypes_memory_stubs.block_address managed))

let string = Ctypes_static.(view (ptr char))
  ~read:string_of_char_ptr ~write:char_ptr_of_string

let read_nullable t reftyp =
  let coerce = Ctypes_coerce.coerce Ctypes_static.(ptr reftyp) t in
  fun p -> if Ctypes_memory.is_null p then None else Some (coerce p)

let write_nullable t reftyp =
  let coerce = Ctypes_coerce.coerce t Ctypes_static.(ptr reftyp) in
  Ctypes_memory.(function None -> from_voidp reftyp null | Some f -> coerce f)

let nullable_view ?format_typ ?format t reftyp =
  let read = read_nullable t reftyp
  and write = write_nullable t reftyp
  in Ctypes_static.(view ~read ~write ?format_typ ?format (ptr reftyp))

let read_nullable_funptr t reftyp =
  let coerce = Ctypes_coerce.coerce (Ctypes_static.static_funptr reftyp) t in
  fun (Ctypes_static.Static_funptr p as ptr) ->
    if Ctypes_ptr.Fat.is_null p
    then None
    else Some (coerce ptr)

let write_nullable_funptr t reftyp =
  let coerce = Ctypes_coerce.coerce t Ctypes_static.(static_funptr reftyp) in
  function None -> Ctypes_static.Static_funptr
                     (Ctypes_ptr.Fat.make ~reftyp Ctypes_ptr.Raw.null)
         | Some f -> coerce f

let nullable_funptr_view ?format_typ ?format t reftyp =
  let read = read_nullable_funptr t reftyp
  and write = write_nullable_funptr t reftyp
  in Ctypes_static.(view ~read ~write ?format_typ ?format (static_funptr reftyp))

let ptr_opt t = nullable_view (Ctypes_static.ptr t) t

let string_opt = nullable_view string Ctypes_static.char

module type Signed_type =
sig
  include Signed.S
  val t : t Ctypes_static.typ
end

module type Unsigned_type =
sig
  include Unsigned.S
  val t : t Ctypes_static.typ
end

let signed_typedef name ~size : (module Signed_type) =
  match size with
    1 -> (module struct include Signed.Int
           let t = Ctypes_static.(typedef int8_t name) end)
  | 2 -> (module struct include Signed.Int
           let t = Ctypes_static.(typedef int16_t name) end)
  | 4 -> (module struct include Signed.Int32
           let t = Ctypes_static.(typedef int32_t name) end)
  | 8 -> (module struct include Signed.Int64
           let t = Ctypes_static.(typedef int64_t name) end)
  | n -> Printf.kprintf failwith "size %d not supported for %s\n" n name

let unsigned_typedef name ~size : (module Unsigned_type) =
  match size with
  | 1 -> (module struct include Unsigned.UInt8
           let t = Ctypes_static.(typedef uint8_t name) end)
  | 2 -> (module struct include Unsigned.UInt16
           let t = Ctypes_static.(typedef uint16_t name) end)
  | 4 -> (module struct include Unsigned.UInt32
           let t = Ctypes_static.(typedef uint32_t name) end)
  | 8 -> (module struct include Unsigned.UInt64
           let t = Ctypes_static.(typedef uint64_t name) end)
  | n -> Printf.kprintf failwith "size %d not supported for %s\n" n name

module Intptr = (val signed_typedef "intptr_t"
                    ~size:(Ctypes_std_view_stubs.intptr_t_size ()))
module Uintptr = (val unsigned_typedef "uintptr_t"
                    ~size:(Ctypes_std_view_stubs.uintptr_t_size ()))
let intptr_t = Intptr.t
let uintptr_t = Uintptr.t

module Ptrdiff = (val signed_typedef "ptrdiff_t"
                     ~size:(Ctypes_std_view_stubs.ptrdiff_t_size ()))
let ptrdiff_t = Ptrdiff.t

end
module Ctypes_structs : sig 
#1 "ctypes_structs.mli"
(*
 * Copyright (c) 2013 Jeremy Yallop.
 *
 * This file is distributed under the terms of the MIT License.
 * See the file LICENSE for details.
 *)

open Ctypes_static

module type S =
sig
  type (_, _) field
  val field : 't typ -> string -> 'a typ ->
    ('a, (('s, [<`Struct | `Union]) structured as 't)) field
  val seal : (_, [< `Struct | `Union]) Ctypes_static.structured Ctypes_static.typ -> unit
end

end = struct
#1 "ctypes_structs.ml"
(*
 * Copyright (c) 2013 Jeremy Yallop.
 *
 * This file is distributed under the terms of the MIT License.
 * See the file LICENSE for details.
 *)

open Ctypes_static

module type S =
sig
  type (_, _) field
  val field : 't typ -> string -> 'a typ ->
    ('a, (('s, [<`Struct | `Union]) structured as 't)) field
  val seal : (_, [< `Struct | `Union]) Ctypes_static.structured Ctypes_static.typ -> unit
end

end
module Ctypes_structs_computed : sig 
#1 "ctypes_structs_computed.mli"
(*
 * Copyright (c) 2013 Jeremy Yallop.
 *
 * This file is distributed under the terms of the MIT License.
 * See the file LICENSE for details.
 *)

(** Structs and unions whose layouts are computed from the sizes and alignment
    requirements of the constituent field types. *)

include Ctypes_structs.S
  with type ('a, 's) field := ('a, 's) Ctypes_static.field

end = struct
#1 "ctypes_structs_computed.ml"
(*
 * Copyright (c) 2013 Jeremy Yallop.
 *
 * This file is distributed under the terms of the MIT License.
 * See the file LICENSE for details.
 *)

open Ctypes_static

let max_field_alignment fields =
  List.fold_left
    (fun align (BoxedField {ftype}) -> max align (alignment ftype))
    0
    fields

let max_field_size fields =
  List.fold_left
    (fun size (BoxedField {ftype}) -> max size (sizeof ftype))
    0
    fields

let aligned_offset offset alignment =
  match offset mod alignment with
    0 -> offset
  | overhang -> offset - overhang + alignment

let rec field : type t a. t typ -> string -> a typ -> (a, t) field =
  fun structured label ftype ->
  match structured with
  | Struct ({ spec = Incomplete spec } as s) ->
    let foffset = aligned_offset spec.isize (alignment ftype) in
    let field = { ftype; foffset; fname = label } in
    begin
      spec.isize <- foffset + sizeof ftype;
      s.fields <- BoxedField field :: s.fields;
      field
    end
  | Union ({ uspec = None } as u) ->
    let field = { ftype; foffset = 0; fname = label } in
    u.ufields <- BoxedField field :: u.ufields;
    field
  | Struct { tag; spec = Complete _ } -> raise (ModifyingSealedType tag)
  | Union { utag } -> raise (ModifyingSealedType utag)
  | View { ty } ->
     let { ftype; foffset; fname } = field ty label ftype in
     { ftype; foffset; fname }
  | _ -> raise (Unsupported "Adding a field to non-structured type")

let rec seal : type a. a typ -> unit = function
  | Struct { fields = [] } -> raise (Unsupported "struct with no fields")
  | Struct { spec = Complete _; tag } -> raise (ModifyingSealedType tag)
  | Struct ({ spec = Incomplete { isize } } as s) ->
    s.fields <- List.rev s.fields;
    let align = max_field_alignment s.fields in
    let size = aligned_offset isize align in
    s.spec <- Complete { (* sraw_io;  *)size; align }
  | Union { utag; uspec = Some _ } ->
    raise (ModifyingSealedType utag)
  | Union { ufields = [] } ->
    raise (Unsupported "union with no fields")
  | Union u -> begin
    u.ufields <- List.rev u.ufields;
    let size = max_field_size u.ufields
    and align = max_field_alignment u.ufields in
    u.uspec <- Some { align; size = aligned_offset size align }
  end
  | View { ty } -> seal ty
  | _ -> raise (Unsupported "Sealing a non-structured type")

end
(** Interface as module  *)
module Ctypes_types
= struct
#1 "ctypes_types.mli"
(*
 * Copyright (c) 2014 Jeremy Yallop.
 *
 * This file is distributed under the terms of the MIT License.
 * See the file LICENSE for details.
 *)

open Signed
open Unsigned


(** Abstract interface to C object type descriptions *)
module type TYPE =
sig
  (** {2:types Values representing C types} *)

  type 'a typ
  (** The type of values representing C types.  There are two types associated
      with each [typ] value: the C type used to store and pass values, and the
      corresponding OCaml type.  The type parameter indicates the OCaml type, so a
      value of type [t typ] is used to read and write OCaml values of type [t].
      There are various uses of [typ] values, including

      - constructing function types for binding native functions using
      {!Foreign.foreign}

      - constructing pointers for reading and writing locations in C-managed
      storage using {!ptr}

      - describing the fields of structured types built with {!structure} and
      {!union}.
  *)

  (** {3 The void type} *)

  val void  : unit typ
  (** Value representing the C void type.  Void values appear in OCaml as the
      unit type, so using void in an argument or result type specification
      produces a function which accepts or returns unit.

      Dereferencing a pointer to void is an error, as in C, and will raise
      {!IncompleteType}.
  *)

  (** {3 Scalar types}

      The scalar types consist of the {!arithmetic_types} and the {!pointer_types}.
  *)

  (** {4:arithmetic_types Arithmetic types}

      The arithmetic types consist of the signed and unsigned integer types
      (including character types) and the floating types.  There are values
      representing both exact-width integer types (of 8, 16, 32 and 64 bits) and
      types whose size depend on the platform (signed and unsigned short, int, long,
      long long).

  *)

  val char : char typ
  (** Value representing the C type [char]. *)

  (** {5 Signed integer types} *)

  val schar : int typ
  (** Value representing the C type [signed char]. *)

  val short : int typ
  (** Value representing the C type ([signed]) [short]. *)

  val int   : int typ
  (** Value representing the C type ([signed]) [int]. *)

  val long  : long typ
  (** Value representing the C type ([signed]) [long]. *)

  val llong  : llong typ
  (** Value representing the C type ([signed]) [long long]. *)

  val nativeint : nativeint typ
  (** Value representing the C type ([signed]) [int]. *)

  val int8_t : int typ
  (** Value representing an 8-bit signed integer C type. *)

  val int16_t : int typ
  (** Value representing a 16-bit signed integer C type. *)

  val int32_t : int32 typ
  (** Value representing a 32-bit signed integer C type. *)

  val int64_t : int64 typ
  (** Value representing a 64-bit signed integer C type. *)

  module Intptr : Signed.S
  val intptr_t : Intptr.t typ
  (** Value representing the C type [intptr_t]. *)

  module Ptrdiff : Signed.S
  val ptrdiff_t : Ptrdiff.t typ
  (** Value representing the C type [ptrdiff_t]. *)

  val camlint : int typ
  (** Value representing an integer type with the same storage requirements as
      an OCaml [int]. *)

  (** {5 Unsigned integer types} *)

  val uchar : uchar typ
  (** Value representing the C type [unsigned char]. *)

  val bool : bool typ
  (** Value representing the C type [bool]. *)

  val uint8_t : uint8 typ
  (** Value representing an 8-bit unsigned integer C type. *)

  val uint16_t : uint16 typ
  (** Value representing a 16-bit unsigned integer C type. *)

  val uint32_t : uint32 typ
  (** Value representing a 32-bit unsigned integer C type. *)

  val uint64_t : uint64 typ
  (** Value representing a 64-bit unsigned integer C type. *)

  val size_t : size_t typ
  (** Value representing the C type [size_t], an alias for one of the unsigned
      integer types.  The actual size and alignment requirements for [size_t]
      vary between platforms. *)

  val ushort : ushort typ
  (** Value representing the C type [unsigned short]. *)

  val sint : sint typ
  (** Value representing the C type [int]. *)

  val uint : uint typ
  (** Value representing the C type [unsigned int]. *)

  val ulong : ulong typ
  (** Value representing the C type [unsigned long]. *)

  val ullong : ullong typ
  (** Value representing the C type [unsigned long long]. *)

  module Uintptr : Unsigned.S
  val uintptr_t : Uintptr.t typ
  (** Value representing the C type [uintptr_t]. *)

  (** {5 Floating types} *)

  val float : float typ
  (** Value representing the C single-precision [float] type. *)

  val double : float typ
  (** Value representing the C type [double]. *)

  (** {5 Complex types} *)

  val complex32 : Complex.t typ
  (** Value representing the C99 single-precision [float complex] type. *)

  val complex64 : Complex.t typ
  (** Value representing the C99 double-precision [double complex] type. *)

  (** {4:pointer_types Pointer types} *)

  (** {5 C-compatible pointers} *)

  val ptr : 'a typ -> 'a Ctypes_static.ptr typ
  (** Construct a pointer type from an existing type (called the {i reference
      type}).  *)

  val ptr_opt : 'a typ -> 'a Ctypes_static.ptr option typ
  (** Construct a pointer type from an existing type (called the {i reference
      type}).  This behaves like {!ptr}, except that null pointers appear in OCaml
      as [None]. *)

  val string : string typ
  (** A high-level representation of the string type.

      On the C side this behaves like [char *]; on the OCaml side values read
      and written using {!string} are simply native OCaml strings.

      To avoid problems with the garbage collector, values passed using
      {!string} are copied into immovable C-managed storage before being passed
      to C.
  *)

  val string_opt : string option typ
  (** A high-level representation of the string type.  This behaves like {!string},
      except that null pointers appear in OCaml as [None].
  *)

  (** {5 OCaml pointers} *)

  val ocaml_string : string Ctypes_static.ocaml typ
  (** Value representing the directly mapped storage of an OCaml string. *)

  val ocaml_bytes : Bytes.t Ctypes_static.ocaml typ
  (** Value representing the directly mapped storage of an OCaml byte array. *)

  (** {3 Array types} *)

  (** {4 C array types} *)

  val array : int -> 'a typ -> 'a Ctypes_static.carray typ
  (** Construct a sized array type from a length and an existing type (called
      the {i element type}). *)

  (** {4 Bigarray types} *)

  val bigarray :
    < element: 'a;
      ba_repr: 'b;
      dims: 'dims;
      bigarray: 'bigarray;
      carray: _ > Ctypes_static.bigarray_class ->
     'dims -> ('a, 'b) Bigarray.kind -> 'bigarray typ
  (** Construct a sized bigarray type representation from a bigarray class, the
      dimensions, and the {!Bigarray.kind}. *)

  val typ_of_bigarray_kind : ('a, 'b) Bigarray.kind -> 'a typ
  (** [typ_of_bigarray_kind k] is the type corresponding to the Bigarray kind
      [k]. *)

  (** {3 Struct and union types} *)

  type ('a, 't) field

  val structure : string -> 's Ctypes_static.structure typ
  (** Construct a new structure type.  The type value returned is incomplete and
      can be updated using {!field} until it is passed to {!seal}, at which point
      the set of fields is fixed.

      The type (['_s structure typ]) of the expression returned by the call
      [structure tag] includes a weak type variable, which can be explicitly
      instantiated to ensure that the OCaml values representing different C
      structure types have incompatible types.  Typical usage is as follows:

      [type tagname]

      [let tagname : tagname structure typ = structure "tagname"]
  *)

  val union : string -> 's Ctypes_static.union typ
  (** Construct a new union type.  This behaves analogously to {!structure};
      fields are added with {!field}. *)

  val field : 't typ -> string -> 'a typ ->
    ('a, (('s, [<`Struct | `Union]) Ctypes_static.structured as 't)) field
  (** [field ty label ty'] adds a field of type [ty'] with label [label] to the
      structure or union type [ty] and returns a field value that can be used to
      read and write the field in structure or union instances (e.g. using
      {!getf} and {!setf}).

      Attempting to add a field to a union type that has been sealed with [seal]
      is an error, and will raise {!ModifyingSealedType}. *)

  val seal : (_, [< `Struct | `Union]) Ctypes_static.structured typ -> unit
  (** [seal t] completes the struct or union type [t] so that no further fields
      can be added.  Struct and union types must be sealed before they can be used
      in a way that involves their size or alignment; see the documentation for
      {!IncompleteType} for further details.  *)

  (** {3 View types} *)

  val view : ?format_typ:((Format.formatter -> unit) -> Format.formatter -> unit) ->
             ?format:(Format.formatter -> 'b -> unit) ->
             read:('a -> 'b) -> write:('b -> 'a) -> 'a typ -> 'b typ
  (** [view ~read:r ~write:w t] creates a C type representation [t'] which
      behaves like [t] except that values read using [t'] are subsequently
      transformed using the function [r] and values written using [t'] are first
      transformed using the function [w].

      For example, given suitable definitions of [string_of_char_ptr] and
      [char_ptr_of_string], the type representation

      [view ~read:string_of_char_ptr ~write:char_ptr_of_string (ptr char)]

      can be used to pass OCaml strings directly to and from bound C functions,
      or to read and write string members in structs and arrays.  (In fact, the
      {!string} type representation is defined in exactly this way.)

      The optional argument [format_typ] is used by the {!Ctypes.format_typ} and
      {!string_of_typ} functions to print the type at the top level and
      elsewhere.  If [format_typ] is not supplied the printer for [t] is used
      instead.

      The optional argument [format] is used by the {!Ctypes.format}
      and {!string_of} functions to print the values. If [format_val]
      is not supplied the printer for [t] is used instead.

  *)

  val typedef : 'a typ -> string -> 'a typ
  (** [typedef t name] creates a C type representation [t'] which
      is equivalent to [t] except its name is printed as [name].

      This is useful when generating C stubs involving "anonymous" types, for
      example: [typedef struct { int f } typedef_name;]
  *)

  (** {3 Abstract types} *)

  val abstract : name:string -> size:int -> alignment:int -> 'a Ctypes_static.abstract typ
  (** Create an abstract type specification from the size and alignment
      requirements for the type. *)

  (** {3 Injection of concrete types} *)

  val lift_typ : 'a Ctypes_static.typ -> 'a typ
  (** [lift_typ t] turns a concrete type representation into an abstract type
      representation.

      For example, retrieving struct layout from C involves working with an
      abstract representation of types which do not support operations such as
      [sizeof].  The [lift_typ] function makes it possible to use concrete
      type representations wherever such abstract type representations are
      needed. *)

  (** {3 Function types} *)
  (** Abstract interface to C function type descriptions *)

  type 'a fn = 'a Ctypes_static.fn
  (** The type of values representing C function types.  A value of type [t fn]
      can be used to bind to C functions and to describe type of OCaml functions
      passed to C. *)

  val ( @-> ) : 'a typ -> 'b fn -> ('a -> 'b) fn
  (** Construct a function type from a type and an existing function type.  This
      corresponds to prepending a parameter to a C function parameter list.  For
      example,

      [int @-> ptr void @-> returning float]

      describes a function type that accepts two arguments -- an integer and a
      pointer to void -- and returns a float.
  *)

  val returning : 'a typ -> 'a fn
  (** Give the return type of a C function.  Note that [returning] is intended
      to be used together with {!(@->)}; see the documentation for {!(@->)} for an
      example. *)

  (** {3 Function pointer types} *)
  type 'a static_funptr = 'a Ctypes_static.static_funptr
  (** The type of values representing C function pointer types. *)

  val static_funptr : 'a fn -> 'a Ctypes_static.static_funptr typ
  (** Construct a function pointer type from an existing function type
      (called the {i reference type}).  *)
end

end
module Ctypes_value_printing_stubs
= struct
#1 "ctypes_value_printing_stubs.ml"
(*
 * Copyright (c) 2013 Jeremy Yallop.
 *
 * This file is distributed under the terms of the MIT License.
 * See the file LICENSE for details.
 *)

(* Stubs for formatting C values. *)

(* Return a string representation of a C value *)
external string_of_prim : 'a Ctypes_primitive_types.prim -> 'a -> string
  = "ctypes_string_of_prim"

external string_of_pointer : _ Ctypes_ptr.Fat.t -> string
  = "ctypes_string_of_pointer"

end
module Ctypes_value_printing
= struct
#1 "ctypes_value_printing.ml"
(*
 * Copyright (c) 2013 Jeremy Yallop.
 *
 * This file is distributed under the terms of the MIT License.
 * See the file LICENSE for details.
 *)

open Ctypes_static
open Ctypes_memory

let rec format : type a. a typ -> Format.formatter -> a -> unit
  = fun typ fmt v -> match typ with
    Void -> Format.pp_print_string fmt ""
  | Primitive p ->
    Format.pp_print_string fmt (Ctypes_value_printing_stubs.string_of_prim p v)
  | Pointer _ -> format_ptr fmt v
  | Funptr _ -> format_funptr fmt v
  | Struct _ -> format_structured fmt v
  | Union _ -> format_structured fmt v
  | Array (a, n) -> format_array fmt v
  | Bigarray ba -> Format.fprintf fmt "<bigarray %a>"
    (fun fmt -> Ctypes_type_printing.format_typ fmt) typ
  | Abstract _ -> format_structured fmt v
  | OCaml _ -> format_ocaml fmt v
  | View {write; ty; format=f} ->
    begin match f with
      | None -> format ty fmt (write v)
      | Some f -> f fmt v
    end
and format_structured : type a b. Format.formatter -> (a, b) structured -> unit
  = fun fmt ({structured = CPointer p} as s) ->
    let open Format in
    match Ctypes_ptr.Fat.reftype p with
    | Struct {fields} ->
      fprintf fmt "{@;<1 2>@[";
      format_fields "," fields fmt s;
      fprintf fmt "@]@;<1 0>}"
    | Union {ufields} ->
      fprintf fmt "{@;<1 2>@[";
      format_fields " |" ufields fmt s;
      fprintf fmt "@]@;<1 0>}"
    | Abstract abs ->
      pp_print_string fmt "<abstract>"
    | _ -> raise (Unsupported "unknown structured type")
and format_array : type a. Format.formatter -> a carray -> unit
  = fun fmt ({astart = CPointer p; alength} as arr) ->
    let open Format in
    fprintf fmt "{@;<1 2>@[";
    for i = 0 to alength - 1 do
      format (Ctypes_ptr.Fat.reftype p) fmt (CArray.get arr i);
      if i <> alength - 1 then
        fprintf fmt ",@;"
    done;
    fprintf fmt "@]@;<1 0>}"
and format_ocaml : type a. Format.formatter -> a ocaml -> unit =
  let offset fmt = function
    | 0 -> ()
    | n -> Format.fprintf fmt "@ @[[offset:%d]@]" n
  and float_array fmt arr =
    Format.fprintf fmt "[|@;<1 2>@[";
    let len = Array.length arr in
    for i = 0 to len - 1 do
      Format.pp_print_float fmt arr.(i);
      if i <> len - 1 then
        Format.fprintf fmt ",@;"
    done;
    Format.fprintf fmt "@]@;<1 0>|]"
  in
  fun fmt (OCamlRef (off, obj, ty)) -> match ty with
  | String -> Format.fprintf fmt "%S%a" obj offset off
  | Bytes -> Format.fprintf fmt "%S%a" (Bytes.to_string obj) offset off
  | FloatArray -> Format.fprintf fmt "%a%a" float_array obj offset off
and format_fields : type a b. string -> (a, b) structured boxed_field list ->
                              Format.formatter -> (a, b) structured -> unit
  = fun sep fields fmt s ->
    let last_field = List.length fields - 1 in
    let open Format in
    List.iteri
      (fun i (BoxedField ({ftype; foffset; fname} as f)) ->
        fprintf fmt "@[%s@] = @[%a@]%s@;" fname (format ftype) (getf s f)
          (if i <> last_field then sep else ""))
      fields
and format_ptr : type a. Format.formatter -> a ptr -> unit
  = fun fmt (CPointer p) ->
    Format.fprintf fmt "%s" (Ctypes_value_printing_stubs.string_of_pointer p)
and format_funptr  : type a. Format.formatter -> a static_funptr -> unit
  = fun fmt (Static_funptr p) ->
    Format.fprintf fmt "%s" (Ctypes_value_printing_stubs.string_of_pointer p)

let string_of typ v = Format.asprintf "%a" (format typ) v

end
module Ctypes : sig 
#1 "ctypes.mli"
(*
 * Copyright (c) 2013 Jeremy Yallop.
 *
 * This file is distributed under the terms of the MIT License.
 * See the file LICENSE for details.
 *)

(** The core ctypes module.

    The main points of interest are the set of functions for describing C
    types (see {!types}) and the set of functions for accessing C values (see
    {!values}).  The {!Foreign.foreign} function uses C type descriptions
    to bind external C values.
*)

(** {4:pointer_types Pointer types} *)

type ('a, 'b) pointer = ('a, 'b) Ctypes_static.pointer
(** The type of pointer values. A value of type [('a, [`C]) pointer] contains
    a C-compatible pointer, and a value of type [('a, [`OCaml]) pointer]
    contains a pointer to a value that can be moved by OCaml runtime. *)

(** {4 C-compatible pointers} *)

type 'a ptr = ('a, [`C]) pointer
(** The type of C-compatible pointer values.  A value of type [t ptr] can be
    used to read and write values of type [t] at particular addresses. *)

type 'a ocaml = 'a Ctypes_static.ocaml
(** The type of pointer values pointing directly into OCaml values.
    {b Pointers of this type should never be captured by external code}.
    In particular, functions accepting ['a ocaml] pointers must not invoke
    any OCaml code. *)

(** {4 C array types} *)

type 'a carray = 'a Ctypes_static.carray
(** The type of C array values.  A value of type [t carray] can be used to read
    and write array objects in C-managed storage. *)

(** {4 Bigarray types} *)

type 'a bigarray_class = 'a Ctypes_static.bigarray_class
(** The type of Bigarray classes.  There are four instances, one for each of
    the Bigarray submodules. *)

val genarray :
  < element: 'a;
    ba_repr: 'b;
    bigarray: ('a, 'b, Bigarray.c_layout) Bigarray.Genarray.t;
    carray: 'a carray;
    dims: int array > bigarray_class
(** The class of {!Bigarray.Genarray.t} values *)

val array1 :
  < element: 'a;
    ba_repr: 'b;
    bigarray: ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t;
    carray: 'a carray;
    dims: int > bigarray_class
(** The class of {!Bigarray.Array1.t} values *)

val array2 :
  < element: 'a;
    ba_repr: 'b;
    bigarray: ('a, 'b, Bigarray.c_layout) Bigarray.Array2.t;
    carray: 'a carray carray;
    dims: int * int > bigarray_class
(** The class of {!Bigarray.Array2.t} values *)

val array3 :
  < element: 'a;
    ba_repr: 'b;
    bigarray: ('a, 'b, Bigarray.c_layout) Bigarray.Array3.t;
    carray: 'a carray carray carray;
    dims: int * int * int > bigarray_class
(** The class of {!Bigarray.Array3.t} values *)

(** {3 Struct and union types} *)

type ('a, 'kind) structured = ('a, 'kind) Ctypes_static.structured
(** The base type of values representing C struct and union types.  The
    ['kind] parameter is a polymorphic variant type indicating whether the type
    represents a struct ([`Struct]) or a union ([`Union]). *)

type 'a structure = ('a, [`Struct]) structured
(** The type of values representing C struct types. *)

type 'a union = ('a, [`Union]) structured
(** The type of values representing C union types. *)

type ('a, 't) field = ('a, 't) Ctypes_static.field
(** The type of values representing C struct or union members (called "fields"
    here).  A value of type [(a, s) field] represents a field of type [a] in a
    struct or union of type [s]. *)

type 'a abstract = 'a Ctypes_static.abstract
(** The type of abstract values.  The purpose of the [abstract] type is to
    represent values whose type varies from platform to platform.

    For example, the type [pthread_t] is a pointer on some platforms, an
    integer on other platforms, and a struct on a third set of platforms.  One
    way to deal with this kind of situation is to have
    possibly-platform-specific code which interrogates the C type in some way
    to help determine an appropriate representation.  Another way is to use
    [abstract], leaving the representation opaque.

    (Note, however, that although [pthread_t] is a convenient example, since
    the type used to implement it varies significantly across platforms, it's
    not actually a good match for [abstract], since values of type [pthread_t]
    are passed and returned by value.) *)

include Ctypes_types.TYPE
 with type 'a typ = 'a Ctypes_static.typ
  and type ('a, 's) field := ('a, 's) field

(** {3 Operations on types} *)

val sizeof : 'a typ -> int
(** [sizeof t] computes the size in bytes of the type [t].  The exception
    {!IncompleteType} is raised if [t] is incomplete. *)

val alignment : 'a typ -> int
(** [alignment t] computes the alignment requirements of the type [t].  The
    exception {!IncompleteType} is raised if [t] is incomplete. *)

val format_typ : ?name:string -> Format.formatter -> 'a typ -> unit
(** Pretty-print a C representation of the type to the specified formatter. *)

val format_fn : ?name:string -> Format.formatter -> 'a fn -> unit
(** Pretty-print a C representation of the function type to the specified
    formatter. *)

val string_of_typ : ?name:string -> 'a typ -> string
(** Return a C representation of the type. *)

val string_of_fn : ?name:string -> 'a fn -> string
(** Return a C representation of the function type. *)

(** {2:values Values representing C values} *)

val format : 'a typ -> Format.formatter -> 'a -> unit
(** Pretty-print a representation of the C value to the specified formatter. *)

val string_of : 'a typ -> 'a -> string
(** Return a string representation of the C value. *)

(** {3 Pointer values} *)

val null : unit ptr
(** A null pointer. *)

val (!@) : 'a ptr -> 'a
(** [!@ p] dereferences the pointer [p].  If the reference type is a scalar
    type then dereferencing constructs a new value.  If the reference type is
    an aggregate type then dereferencing returns a value that references the
    memory pointed to by [p]. *)

val (<-@) : 'a ptr -> 'a -> unit
(** [p <-@ v] writes the value [v] to the address [p]. *)

val (+@) : ('a, 'b) pointer -> int -> ('a, 'b) pointer
(** If [p] is a pointer to an array element then [p +@ n] computes the
    address of the [n]th next element. *)

val (-@) : ('a, 'b) pointer -> int -> ('a, 'b) pointer
(** If [p] is a pointer to an array element then [p -@ n] computes the address
    of the nth previous element. *)

val ptr_diff : ('a, 'b) pointer -> ('a, 'b) pointer -> int
(** [ptr_diff p q] computes [q - p].  As in C, both [p] and [q] must point
    into the same array, and the result value is the difference of the
    subscripts of the two array elements. *)

val from_voidp : 'a typ -> unit ptr -> 'a ptr
(** Conversion from [void *]. *)

val to_voidp : _ ptr -> unit ptr
(** Conversion to [void *]. *)

val allocate : ?finalise:('a ptr -> unit) -> 'a typ -> 'a -> 'a ptr
(** [allocate t v] allocates a fresh value of type [t], initialises it
    with [v] and returns its address.  The argument [?finalise], if
    present, will be called just before the memory is freed.  The value
    will be automatically freed after no references to the pointer
    remain within the calling OCaml program. *)

val allocate_n : ?finalise:('a ptr -> unit) -> 'a typ -> count:int -> 'a ptr
(** [allocate_n t ~count:n] allocates a fresh array with element type
    [t] and length [n], and returns its address.  The argument
    [?finalise], if present, will be called just before the memory is
    freed.  The array will be automatically freed after no references
    to the pointer remain within the calling OCaml program.  The
    memory is allocated with libc's [calloc] and is guaranteed to be
    zero-filled.  *)

val ptr_compare : 'a ptr -> 'a ptr -> int
(** If [p] and [q] are pointers to elements [i] and [j] of the same array then
    [ptr_compare p q] compares the indexes of the elements.  The result is
    negative if [i] is less than [j], positive if [i] is greater than [j], and
    zero if [i] and [j] are equal. *)

val reference_type : 'a ptr -> 'a typ
(** Retrieve the reference type of a pointer. *)

val ptr_of_raw_address : nativeint -> unit ptr
(** Convert the numeric representation of an address to a pointer *)

val funptr_of_raw_address : nativeint -> (unit -> unit) Ctypes_static.static_funptr
(** Convert the numeric representation of an address to a function pointer *)

val raw_address_of_ptr : unit ptr -> nativeint
(** [raw_address_of_ptr p] returns the numeric representation of p.

    Note that the return value remains valid only as long as the pointed-to
    object is alive.  If [p] is a managed object (e.g. a value returned by
    {!make}) then unless the caller retains a reference to [p], the object may
    be collected, invalidating the returned address. *)

val string_from_ptr : char ptr -> length:int -> string
(** [string_from_ptr p ~length] creates a string initialized with the [length]
    characters at address [p].

    Raise [Invalid_argument "Ctypes.string_from_ptr"] if [length] is
    negative. *)

val ocaml_string_start : string -> string ocaml
(** [ocaml_string_start s] allows to pass a pointer to the contents of an OCaml
    string directly to a C function. *)

val ocaml_bytes_start : Bytes.t -> Bytes.t ocaml
(** [ocaml_bytes_start s] allows to pass a pointer to the contents of an OCaml
    byte array directly to a C function. *)

(** {3 Array values} *)

(** {4 C array values} *)

module CArray :
sig
  type 'a t = 'a carray

  val get : 'a t -> int -> 'a
  (** [get a n] returns the [n]th element of the zero-indexed array [a].  The
      semantics for non-scalar types are non-copying, as for {!(!@)}.

      If you rebind the [CArray] module to [Array] then you can also use the
      syntax [a.(n)] instead of [Array.get a n].

      Raise [Invalid_argument "index out of bounds"] if [n] is outside of the
      range [0] to [(CArray.length a - 1)]. *)

  val set : 'a t -> int -> 'a -> unit
  (** [set a n v] overwrites the [n]th element of the zero-indexed array [a]
      with [v].

      If you rebind the [CArray] module to [Array] then you can also use the
      [a.(n) <- v] syntax instead of [Array.set a n v].

      Raise [Invalid_argument "index out of bounds"] if [n] is outside of the
      range [0] to [(CArray.length a - 1)]. *)

  val unsafe_get : 'a t -> int -> 'a
  (** [unsafe_get a n] behaves like [get a n] except that the check that [n]
      between [0] and [(CArray.length a - 1)] is not performed. *)

  val unsafe_set : 'a t -> int -> 'a -> unit
  (** [unsafe_set a n v] behaves like [set a n v] except that the check that
      [n] between [0] and [(CArray.length a - 1)] is not performed. *)

  val of_list : 'a typ -> 'a list -> 'a t
  (** [of_list t l] builds an array of type [t] of the same length as [l], and
      writes the elements of [l] to the corresponding elements of the array. *)

  val to_list : 'a t -> 'a list
  (** [to_list a] builds a list of the same length as [a] such that each
      element of the list is the result of reading the corresponding element of
      [a]. *)

  val length : 'a t -> int
  (** Return the number of elements of the given array. *)

  val start : 'a t -> 'a ptr
  (** Return the address of the first element of the given array. *)

  val from_ptr : 'a ptr -> int -> 'a t
  (** [from_ptr p n] creates an [n]-length array reference to the memory at
      address [p]. *)

  val make : ?finalise:('a t -> unit) -> 'a typ -> ?initial:'a -> int -> 'a t
  (** [make t n] creates an [n]-length array of type [t].  If the optional
      argument [?initial] is supplied, it indicates a value that should be
      used to initialise every element of the array.  The argument [?finalise],
      if present, will be called just before the memory is freed. *)

  val element_type : 'a t -> 'a typ
(** Retrieve the element type of an array. *)
end
(** Operations on C arrays. *)

(** {4 Bigarray values} *)

val bigarray_start : < element: 'a;
                       ba_repr: _;
                       bigarray: 'b;
                       carray: _;
                       dims: _ > bigarray_class -> 'b -> 'a ptr
(** Return the address of the first element of the given Bigarray value. *)

val bigarray_of_ptr : < element: 'a;
                        ba_repr: 'f;
                        bigarray: 'b;
                        carray: _;
                        dims: 'i > bigarray_class ->
    'i -> ('a, 'f) Bigarray.kind -> 'a ptr -> 'b
(** [bigarray_of_ptr c dims k p] converts the C pointer [p] to a bigarray
    value.  No copy is made; the bigarray references the memory pointed to by
    [p]. *)

val array_of_bigarray : < element: _;
                          ba_repr: _;
                          bigarray: 'b;
                          carray: 'c;
                          dims: _ > bigarray_class -> 'b -> 'c
(** [array_of_bigarray c b] converts the bigarray value [b] to a value of type
    {!CArray.t}.  No copy is made; the result occupies the same memory as
    [b]. *)

(** Convert a Bigarray value to a C array. *)

val bigarray_of_array : < element: 'a;
                          ba_repr: 'f;
                          bigarray: 'b;
                          carray: 'c carray;
                          dims: 'i > bigarray_class ->
    ('a, 'f) Bigarray.kind -> 'c carray -> 'b
(** [bigarray_of_array c k a] converts the {!CArray.t} value [a] to a bigarray
    value.  No copy is made; the result occupies the same memory as [a]. *)

(** {3 Struct and union values} *)

val make : ?finalise:('s -> unit) -> ((_, _) structured as 's) typ -> 's
(** Allocate a fresh, uninitialised structure or union value.  The argument
    [?finalise], if present, will be called just before the underlying memory is
    freed. *)

val setf : ((_, _) structured as 's) -> ('a, 's) field -> 'a -> unit
(** [setf s f v] overwrites the value of the field [f] in the structure or
    union [s] with [v]. *)

val getf : ((_, _) structured as 's) -> ('a, 's) field -> 'a
(** [getf s f] retrieves the value of the field [f] in the structure or union
    [s].  The semantics for non-scalar types are non-copying, as for
    {!(!@)}.*)

val (@.) : ((_, _) structured as 's) -> ('a, 's) field -> 'a ptr
(** [s @. f] computes the address of the field [f] in the structure or union
    value [s]. *)

val (|->) : ((_, _) structured as 's) ptr -> ('a, 's) field -> 'a ptr
(** [p |-> f] computes the address of the field [f] in the structure or union
    value pointed to by [p]. *)

val offsetof : (_, _ structure) field -> int
(** [offsetof f] returns the offset, in bytes, of the field [f] from the
    beginning of the associated struct type. *)

val field_type : ('a, _) field -> 'a typ
(** [field_type f] returns the type of the field [f]. *)

val field_name : (_, _) field -> string
(** [field_name f] returns the name of the field [f]. *)

val addr : ((_, _) structured as 's) -> 's ptr
(** [addr s] returns the address of the structure or union [s]. *)

(** {3 Coercions} *)

val coerce : 'a typ -> 'b typ -> 'a -> 'b
(** [coerce t1 t2] returns a coercion function between the types represented
    by [t1] and [t2].  If [t1] cannot be coerced to [t2], [coerce] raises
    {!Uncoercible}.

    The following coercions are currently supported:

     - All function and object pointer types are intercoercible.
     - Any type may be coerced to {!void}
     - There is a coercion between a {!view} and another type [t] (in either
       direction) if there is a coercion between the representation type
       underlying the view and [t].
     - Coercion is transitive: if [t1] is coercible to [t2] and [t2] is
       coercible to [t3], then [t1] is directly coercible to [t3].

    The set of supported coercions is subject to change.  Future versions of
    ctypes may both add new types of coercion and restrict the existing
    coercions. *)

val coerce_fn : 'a fn -> 'b fn -> 'a -> 'b
(** [coerce_fn f1 f2] returns a coercion function between the function
    types represented by [f1] and [f2].  If [f1] cannot be coerced to
    [f2], [coerce_fn] raises {!Uncoercible}.

    A function type [f1] may be coerced to another function type [f2]
    if all of the following hold:

      - the C types described by [f1] and [f2] have the same arity

      - each argument of [f2] may be coerced to the corresponding
        argument of [f1]

      - the return type of [f1] may be coerced to the return type of [f2]

    The set of supported coercions is subject to change.  Future versions of
    ctypes may both add new types of coercion and restrict the existing
    coercions. *)

(** {2:roots Registration of OCaml values as roots} *)
module Root :
sig
  val create : 'a -> unit ptr
  (** [create v] allocates storage for the address of the OCaml value [v],
      registers the storage as a root, and returns its address. *)

  val get : unit ptr -> 'a
  (** [get p] retrieves the OCaml value whose address is stored at [p]. *)

  val set : unit ptr -> 'a -> unit
  (** [set p v] updates the OCaml value stored as a root at [p]. *)

  val release : unit ptr -> unit
  (** [release p] unregsiters the root [p]. *)
end

(** {2 Exceptions} *)

exception Unsupported of string
(** An attempt was made to use a feature not currently supported by ctypes.
    In practice this refers to attempts to use an union, array or abstract
    type as an argument or return type of a function. *)

exception ModifyingSealedType of string
(** An attempt was made to modify a sealed struct or union type
    description.  *)

exception IncompleteType
(** An attempt was made to compute the size or alignment of an incomplete
    type.

    The incomplete types are struct and union types that have not been sealed,
    and the void type.

    It is not permitted to compute the size or alignment requirements of an
    incomplete type, to use it as a struct or union member, to read or write a
    value of the type through a pointer or to use it as the referenced type in
    pointer arithmetic.  Additionally, incomplete struct and union types
    cannot be used as argument or return types.
*)

type uncoercible_info
exception Uncoercible of uncoercible_info
(** An attempt was made to coerce between uncoercible types.  *)

end = struct
#1 "ctypes.ml"
(*
 * Copyright (c) 2013 Jeremy Yallop.
 *
 * This file is distributed under the terms of the MIT License.
 * See the file LICENSE for details.
 *)

include Ctypes_static

include Ctypes_structs_computed

include Ctypes_type_printing

include Ctypes_memory

include Ctypes_std_views

include Ctypes_value_printing

include Ctypes_coerce

let lift_typ x = x

end
module Ctypes_closure_properties : sig 
#1 "ctypes_closure_properties.mli"
(*
 * Copyright (c) 2013 Jeremy Yallop.
 *
 * This file is distributed under the terms of the MIT License.
 * See the file LICENSE for details.
 *)

module type MUTEX =
sig
  type t
  val create : unit -> t
  val lock : t -> unit
  val try_lock : t -> bool
  val unlock : t -> unit
end

module Make (Mutex : MUTEX) :
sig
  val record : Obj.t -> Obj.t -> int
  (** [record c v] links the lifetimes of [c] and [v], ensuring that [v] is not
      collected while [c] is still live.  The return value is a key
      that can be used to retrieve [v] while [v] is still live. *)

  val retrieve : int -> Obj.t
  (** [retrieve v] retrieves a value using a key returned by [record], or raises
      [Not_found] if [v] is no longer live. *)
end

end = struct
#1 "ctypes_closure_properties.ml"
(*
 * Copyright (c) 2013 Jeremy Yallop.
 *
 * This file is distributed under the terms of the MIT License.
 * See the file LICENSE for details.
 *)

module type MUTEX =
sig
  type t
  val create : unit -> t
  val lock : t -> unit
  val try_lock : t -> bool
  val unlock : t -> unit
end

module HashPhysical = Hashtbl.Make
  (struct
    type t = Obj.t
    let hash = Hashtbl.hash
    let equal = (==)
   end)

module Make (Mutex : MUTEX) = struct

  (* Map integer identifiers to functions. *)
  let function_by_id : (int, Obj.t) Hashtbl.t = Hashtbl.create 10

  (* Map functions (not closures) to identifiers. *)
  let id_by_function : int HashPhysical.t = HashPhysical.create 10

  (* A single mutex guards both tables *)
  let tables_lock = Mutex.create ()

  (* (The caller must hold tables_lock) *)
  let store_non_closure_function fn boxed_fn id =
    try
      (* Return the existing identifier, if any. *)
      HashPhysical.find id_by_function fn
    with Not_found ->
      (* Add entries to both tables *)
      HashPhysical.add id_by_function fn id;
      Hashtbl.add function_by_id id boxed_fn;
      id

  let fresh () = Oo.id (object end)

  let finalise key =
    (* GC can be triggered while the lock is already held, in which case we
       abandon the attempt and re-install the finaliser. *)
    let rec cleanup fn =
      begin
        if Mutex.try_lock tables_lock then begin
          Hashtbl.remove function_by_id key;
          Mutex.unlock tables_lock;
        end
        else Gc.finalise cleanup fn;
      end
    in cleanup

  let record closure boxed_closure : int =
    let key = fresh () in
    try
      (* For closures we add an entry to function_by_id and a finaliser that
         removes the entry. *)
      Gc.finalise (finalise key) closure;
      begin
        Mutex.lock tables_lock;
        Hashtbl.add function_by_id key boxed_closure;
        Mutex.unlock tables_lock;
      end;
      key
    with Invalid_argument "Gc.finalise" ->
      (* For non-closures we add entries to function_by_id and
         id_by_function. *)
      begin
        Mutex.lock tables_lock;
        let id = store_non_closure_function closure boxed_closure key in
        Mutex.unlock tables_lock;
        id
      end

  let retrieve id =
    begin
      Mutex.lock tables_lock;
      let f =
        try Hashtbl.find function_by_id id
        with Not_found ->
          Mutex.unlock tables_lock;
          raise Not_found
      in begin
        Mutex.unlock tables_lock;
        f
      end
    end
end

end
module Ctypes_ffi_stubs
= struct
#1 "ctypes_ffi_stubs.ml"
(*
 * Copyright (c) 2013 Jeremy Yallop.
 *
 * This file is distributed under the terms of the MIT License.
 * See the file LICENSE for details.
 *)

(* Stubs for binding to libffi. *)

open Ctypes_ptr

(* The type of structure types *)
type 'a ffitype = voidp
type struct_ffitype

external primitive_ffitype : 'a Ctypes_primitive_types.prim -> 'a ffitype
 = "ctypes_primitive_ffitype"

external pointer_ffitype : unit -> voidp ffitype
 = "ctypes_pointer_ffitype"

external void_ffitype : unit -> unit ffitype
 = "ctypes_void_ffitype"


(* Allocate a new C typed buffer specification *)
external allocate_struct_ffitype : int -> struct_ffitype
  = "ctypes_allocate_struct_ffitype"

external struct_type_set_argument : struct_ffitype -> int -> _ ffitype -> unit
  = "ctypes_struct_ffitype_set_argument"

(* Produce a structure type representation from the buffer specification. *)
external complete_struct_type : struct_ffitype -> unit
  = "ctypes_complete_structspec"

external ffi_type_of_struct_type : struct_ffitype -> _ ffitype
  = "ctypes_block_address"

(* A specification of argument C-types and C-return values *)
type callspec

(* Allocate a new C call specification *)
external allocate_callspec : check_errno:bool -> runtime_lock:bool ->
  thread_registration:bool -> callspec
  = "ctypes_allocate_callspec"

(* Add an argument to the C buffer specification *)
external add_argument : callspec -> _ ffitype -> int
  = "ctypes_add_argument"

(* Pass the return type and conclude the specification preparation *)
external prep_callspec : callspec -> int -> _ ffitype -> unit
  = "ctypes_prep_callspec"

(* Call the function specified by `callspec' at the given address.
   The callback functions write the arguments to the buffer and read
   the return value. *)
external call : string -> _ Ctypes_static.fn Fat.t -> callspec ->
  (voidp -> (Obj.t * int) array -> unit) -> (voidp -> 'a) -> 'a
  = "ctypes_call"


(* nary callbacks *)
type boxedfn =
  | Done of (voidp -> unit) * callspec
  | Fn of (voidp -> boxedfn)

type funptr_handle

(* Construct a pointer to an OCaml function represented by an identifier *)
external make_function_pointer : callspec -> int -> funptr_handle
  = "ctypes_make_function_pointer"

external raw_address_of_function_pointer : funptr_handle -> voidp
  = "ctypes_raw_address_of_function_pointer"

(* Set the function used to retrieve functions by identifier. *)
external set_closure_callback : (int -> Obj.t) -> unit
  = "ctypes_set_closure_callback"


(* An internal error: for example, an `ffi_type' object passed to ffi_prep_cif
   was incorrect. *)
exception Ffi_internal_error of string
let () = Callback.register_exception "FFI_internal_error"
  (Ffi_internal_error "")

(* A closure passed to C was collected by the OCaml garbage collector before
   it was called. *)
exception CallToExpiredClosure
let () = Callback.register_exception "CallToExpiredClosure"
  CallToExpiredClosure

end
module Ctypes_weak_ref : sig 
#1 "ctypes_weak_ref.mli"
(*
 * Copyright (c) 2013 Jeremy Yallop.
 *
 * This file is distributed under the terms of the MIT License.
 * See the file LICENSE for details.
 *)

(** A single-cell variant of the weak arrays in the standard library. *)

exception EmptyWeakReference
(** An expired weak reference was accessed. *)

type 'a t
(** The type of weak references.. *)

val make : 'a -> 'a t
(** Obtain a weak reference from a strong reference. *)

val set : 'a t -> 'a -> unit
(** Update a weak reference. *)

val get : 'a t -> 'a
(** Obtain a strong reference from a weak reference. *)

val is_empty : 'a t -> bool
(** Whether a weak reference is still live. *)

end = struct
#1 "ctypes_weak_ref.ml"
(*
 * Copyright (c) 2013 Jeremy Yallop.
 *
 * This file is distributed under the terms of the MIT License.
 * See the file LICENSE for details.
 *)

exception EmptyWeakReference

type 'a t = 'a Weak.t

let empty () = raise EmptyWeakReference
let make v = Weak.(let a = create 1 in set a 0 (Some v); a)
let set r v = Weak.set r 0 (Some v)
let get r = match Weak.get r 0 with Some v -> v | None -> empty ()
let is_empty r = Weak.check r 0

end
module Libffi_abi : sig 
#1 "libffi_abi.mli"
(*
 * Copyright (c) 2014 Jeremy Yallop.
 *
 * This file is distributed under the terms of the MIT License.
 * See the file LICENSE for details.
 *)

(** Support for various ABIs. *)

type abi

val aix : abi
val darwin : abi
val eabi : abi
val fastcall : abi
val gcc_sysv : abi
val linux : abi
val linux64 : abi
val linux_soft_float : abi
val ms_cdecl : abi
val n32 : abi
val n32_soft_float : abi
val n64 : abi
val n64_soft_float : abi
val o32 : abi
val o32_soft_float : abi
val osf : abi
val pa32 : abi
val stdcall : abi
val sysv : abi
val thiscall : abi
val unix : abi
val unix64 : abi
val v8 : abi
val v8plus : abi
val v9 : abi
val vfp : abi

val default_abi : abi

val abi_code : abi -> int

end = struct
#1 "libffi_abi.ml"
(*
 * Copyright (c) 2014 Jeremy Yallop.
 *
 * This file is distributed under the terms of the MIT License.
 * See the file LICENSE for details.
 *)

(* Support for various ABIs *)

type abi = Code of int | Unsupported of string

let abi_code = function
   Code c -> c
 | Unsupported sym -> raise (Ctypes.Unsupported sym)

let aix = Unsupported "FFI_AIX"
let darwin = Unsupported "FFI_DARWIN"
let eabi = Unsupported "FFI_EABI"
let fastcall = Code 4
let gcc_sysv = Unsupported "FFI_GCC_SYSV"
let linux = Unsupported "FFI_LINUX"
let linux64 = Unsupported "FFI_LINUX64"
let linux_soft_float = Unsupported "FFI_LINUX_SOFT_FLOAT"
let ms_cdecl = Unsupported "FFI_MS_CDECL"
let n32 = Unsupported "FFI_N32"
let n32_soft_float = Unsupported "FFI_N32_SOFT_FLOAT"
let n64 = Unsupported "FFI_N64"
let n64_soft_float = Unsupported "FFI_N64_SOFT_FLOAT"
let o32 = Unsupported "FFI_O32"
let o32_soft_float = Unsupported "FFI_O32_SOFT_FLOAT"
let osf = Unsupported "FFI_OSF"
let pa32 = Unsupported "FFI_PA32"
let stdcall = Code 5
let sysv = Code 1
let thiscall = Code 3
let unix = Unsupported "FFI_UNIX"
let unix64 = Code 2
let v8 = Unsupported "FFI_V8"
let v8plus = Unsupported "FFI_V8PLUS"
let v9 = Unsupported "FFI_V9"
let vfp = Unsupported "FFI_VFP"
let default_abi = Code 2

end
module Ctypes_ffi : sig 
#1 "ctypes_ffi.mli"
(*
 * Copyright (c) 2013 Jeremy Yallop.
 *
 * This file is distributed under the terms of the MIT License.
 * See the file LICENSE for details.
 *)

module type CLOSURE_PROPERTIES =
sig
  val record : Obj.t -> Obj.t -> int
  (** [record c v] links the lifetimes of [c] and [v], ensuring that [v] is not
      collected while [c] is still live.  The return value is a key
      that can be used to retrieve [v] while [v] is still live. *)

  val retrieve : int -> Obj.t
  (** [retrieve v] retrieves a value using a key returned by [record], or raises
      [Not_found] if [v] is no longer live. *)
end

module Make(Closure_properties : CLOSURE_PROPERTIES) :
sig
  open Ctypes_static
  open Libffi_abi

  (** Dynamic function calls based on libffi *)

  val function_of_pointer : ?name:string -> abi:abi -> check_errno:bool ->
    release_runtime_lock:bool -> ('a -> 'b) fn -> ('a -> 'b) static_funptr ->
    ('a -> 'b)
  (** Build an OCaml function from a type specification and a pointer to a C
      function. *)

  val pointer_of_function : abi:abi -> acquire_runtime_lock:bool ->
    thread_registration:bool ->
    ('a -> 'b) fn -> ('a -> 'b) -> ('a -> 'b) static_funptr
  (** Build an C function from a type specification and an OCaml function.

      The C function pointer returned is callable as long as the OCaml function
      value is live. *)
end

end = struct
#1 "ctypes_ffi.ml"
(*
 * Copyright (c) 2013 Jeremy Yallop.
 *
 * This file is distributed under the terms of the MIT License.
 * See the file LICENSE for details.
 *)

module type CLOSURE_PROPERTIES =
sig
  val record : Obj.t -> Obj.t -> int
  (** [record c v] links the lifetimes of [c] and [v], ensuring that [v] is not
      collected while [c] is still live.  The return value is a key
      that can be used to retrieve [v] while [v] is still live. *)

  val retrieve : int -> Obj.t
  (** [retrieve v] retrieves a value using a key returned by [record], or raises
      [Not_found] if [v] is no longer live. *)
end

module Make(Closure_properties : CLOSURE_PROPERTIES) =
struct

  open Ctypes_static
  open Libffi_abi

  (* Register the closure lookup function with C. *)
  let () = Ctypes_ffi_stubs.set_closure_callback Closure_properties.retrieve

  type _ ccallspec =
      Call : bool * (Ctypes_ptr.voidp -> 'a) -> 'a ccallspec
    | WriteArg : ('a -> Ctypes_ptr.voidp -> (Obj.t * int) array -> unit) * 'b ccallspec ->
                 ('a -> 'b) ccallspec

  type arg_type = ArgType : 'a Ctypes_ffi_stubs.ffitype -> arg_type

  (* keep_alive ties the lifetimes of objects together.

     [keep_alive w ~while_live:v] ensures that [w] is not collected while [v] is
     still live.

     If the object v in the call [keep_alive w ~while_live:v] is
     static -- for example, if it is a top-level function -- then it
     is not possible to attach a finaliser to [v] and [w] should be
     kept alive indefinitely, which we achieve by adding it to the
     list [kept_alive_indefinitely].
  *)
  let kept_alive_indefinitely = ref []
  let keep_alive w ~while_live:v =
    try Gc.finalise (fun _ -> Ctypes_memory_stubs.use_value w; ()) v
    with Invalid_argument "Gc.finalise" ->
      kept_alive_indefinitely := Obj.repr w :: !kept_alive_indefinitely

  let report_unpassable what =
    let msg = Printf.sprintf "libffi does not support passing %s" what in
    raise (Unsupported msg)

  let rec arg_type : type a. a typ -> arg_type = function
    | Void                                -> ArgType (Ctypes_ffi_stubs.void_ffitype ())
    | Primitive p as prim                 -> let ffitype = Ctypes_ffi_stubs.primitive_ffitype p in
                                             if ffitype = Ctypes_ptr.Raw.null
                                             then report_unpassable
                                               (Ctypes_type_printing.string_of_typ prim)
                                             else ArgType ffitype
    | Pointer _                           -> ArgType (Ctypes_ffi_stubs.pointer_ffitype ())
    | Funptr _                            -> ArgType (Ctypes_ffi_stubs.pointer_ffitype ())
    | OCaml _                             -> ArgType (Ctypes_ffi_stubs.pointer_ffitype ())
    | Union _                             -> report_unpassable "unions"
    | Struct ({ spec = Complete _ } as s) -> struct_arg_type s
    | View { ty }                         -> arg_type ty
    | Array _                             -> report_unpassable "arrays"
    | Bigarray _                          -> report_unpassable "bigarrays"
    | Abstract _                          -> (report_unpassable
                                                "values of abstract type")
    (* The following case should never happen; incomplete types are excluded
       during type construction. *)
    | Struct { spec = Incomplete _ }      -> report_unpassable "incomplete types"
  and struct_arg_type : type s. s structure_type -> arg_type =
     fun ({fields} as s) ->
       let bufspec = Ctypes_ffi_stubs.allocate_struct_ffitype (List.length fields) in
       (* Ensure that `bufspec' stays alive as long as the type does. *)
       keep_alive bufspec ~while_live:s;
       List.iteri
         (fun i (BoxedField {ftype; foffset}) ->
           let ArgType t = arg_type ftype in
           Ctypes_ffi_stubs.struct_type_set_argument bufspec i t)
         fields;
       Ctypes_ffi_stubs.complete_struct_type bufspec;
       ArgType (Ctypes_ffi_stubs.ffi_type_of_struct_type bufspec)

  (*
    call addr callspec
     (fun buffer ->
          write arg_1 buffer v_1
          write arg buffer v
          ...
          write arg_n buffer v_n)
     read_return_value
  *)
  let rec invoke : type a b.
    string option ->
    a ccallspec ->
    (Ctypes_ptr.voidp -> (Obj.t * int) array -> unit) list ->
    Ctypes_ffi_stubs.callspec ->
    b fn Ctypes_ptr.Fat.t ->
    a
    = fun name -> function
      | Call (check_errno, read_return_value) ->
        let name = match name with Some name -> name | None -> "" in
        fun writers callspec addr ->
          Ctypes_ffi_stubs.call name addr callspec
            (fun buf arr -> List.iter (fun w -> w buf arr) writers)
            read_return_value
      | WriteArg (write, ccallspec) ->
        let next = invoke name ccallspec in
        fun writers callspec addr v ->
          next (write v :: writers) callspec addr

  let add_argument : type a. Ctypes_ffi_stubs.callspec -> a typ -> int
    = fun callspec -> function
      | Void -> 0
      | ty   -> let ArgType ffitype = arg_type ty in
                Ctypes_ffi_stubs.add_argument callspec ffitype

  let prep_callspec callspec abi ty =
    let ArgType ctype = arg_type ty in
    Ctypes_ffi_stubs.prep_callspec callspec (abi_code abi) ctype

  let rec box_function : type a. abi -> a fn -> Ctypes_ffi_stubs.callspec -> a Ctypes_weak_ref.t ->
      Ctypes_ffi_stubs.boxedfn
    = fun abi fn callspec -> match fn with
      | Returns ty ->
        let () = prep_callspec callspec abi ty in
        let write_rv = Ctypes_memory.write ty in
        fun f ->
          let w = write_rv (Ctypes_weak_ref.get f) in
          Ctypes_ffi_stubs.Done ((fun p -> w (Ctypes_ptr.Fat.make ~reftyp:Void p)),
                          callspec)
      | Function (p, f) ->
        let _ = add_argument callspec p in
        let box = box_function abi f callspec in
        let read = Ctypes_memory.build p in
        fun f -> Ctypes_ffi_stubs.Fn (fun buf ->
          let f' =
            try Ctypes_weak_ref.get f (read (Ctypes_ptr.Fat.make ~reftyp:Void buf))
            with Ctypes_weak_ref.EmptyWeakReference ->
              raise Ctypes_ffi_stubs.CallToExpiredClosure
          in
          let v = box (Ctypes_weak_ref.make f') in
          let () = Gc.finalise (fun _ -> Ctypes_memory_stubs.use_value f') v in
          v)

  let write_arg : type a. a typ -> offset:int -> idx:int -> a ->
                  Ctypes_ptr.voidp -> (Obj.t * int) array -> unit =
    let ocaml_arg elt_size =
      fun ~offset ~idx (OCamlRef (disp, obj, _)) dst mov ->
        mov.(idx) <- (Obj.repr obj, disp * elt_size)
    in function
    | OCaml String     -> ocaml_arg 1
    | OCaml Bytes      -> ocaml_arg 1
    | OCaml FloatArray -> ocaml_arg (Ctypes_primitives.sizeof Ctypes_primitive_types.Double)
    | ty -> (fun ~offset ~idx v dst mov -> Ctypes_memory.write ty v
      (Ctypes_ptr.Fat.(add_bytes (make ~reftyp:Void dst) offset)))

  (*
    callspec = allocate_callspec ()
    add_argument callspec arg1
    add_argument callspec arg2
    ...
    add_argument callspec argn
    prep_callspec callspec rettype
  *)
  let rec build_ccallspec : type a. abi:abi -> check_errno:bool -> ?idx:int -> a fn ->
    Ctypes_ffi_stubs.callspec -> a ccallspec
    = fun ~abi ~check_errno ?(idx=0) fn callspec -> match fn with
      | Returns t ->
        let () = prep_callspec callspec abi t in
        let b = Ctypes_memory.build t in
        Call (check_errno, (fun p -> b (Ctypes_ptr.Fat.make ~reftyp:Void p)))
      | Function (p, f) ->
        let offset = add_argument callspec p in
        let rest = build_ccallspec ~abi ~check_errno ~idx:(idx+1) f callspec in
        WriteArg (write_arg p ~offset ~idx, rest)

  let build_function ?name ~abi ~release_runtime_lock ~check_errno fn =
    let c = Ctypes_ffi_stubs.allocate_callspec ~check_errno
      ~runtime_lock:release_runtime_lock
      ~thread_registration:false
    in
    let e = build_ccallspec ~abi ~check_errno fn c in
    invoke name e [] c

  let funptr_of_rawptr fn raw_ptr =
    Static_funptr (Ctypes_ptr.Fat.make ~reftyp:fn raw_ptr)

  let function_of_pointer ?name ~abi ~check_errno ~release_runtime_lock fn =
    if release_runtime_lock && has_ocaml_argument fn
    then raise (Unsupported "Unsupported argument type when releasing runtime lock")
    else
      let f = build_function ?name ~abi ~check_errno ~release_runtime_lock fn in
      fun (Static_funptr p) -> f p

  let pointer_of_function ~abi ~acquire_runtime_lock ~thread_registration fn =
    let cs' = Ctypes_ffi_stubs.allocate_callspec
      ~check_errno:false
      ~runtime_lock:acquire_runtime_lock
      ~thread_registration
    in
    let cs = box_function abi fn cs' in
    fun f ->
      let boxed = cs (Ctypes_weak_ref.make f) in
      let id = Closure_properties.record (Obj.repr f) (Obj.repr boxed) in
      let funptr = Ctypes_ffi_stubs.make_function_pointer cs' id in
      (* TODO: use a more intelligent strategy for keeping function pointers
         associated with top-level functions alive (e.g. cache function
         pointer creation by (function, type), or possibly even just by
         function, since the C arity and types must be the same in each case.)
         See the note by [kept_alive_indefinitely].  *)
      let () = keep_alive funptr ~while_live:f in
      funptr_of_rawptr fn
        (Ctypes_ffi_stubs.raw_address_of_function_pointer funptr)
end

end
module Dl : sig 
#1 "dl.mli"
(*
 * Copyright (c) 2013 Jeremy Yallop.
 *
 * This file is distributed under the terms of the MIT License.
 * See the file LICENSE for details.
 *)

(** Bindings to the dlopen / dlsym interface. *)

type library
(** The type of dynamic libraries, as returned by {!dlopen}. *)

exception DL_error of string
(** An error condition occurred when calling {!dlopen}, {!dlclose} or
    {!dlsym}.  The argument is the string returned by the [dlerror]
    function. *)

(** Flags for {!dlopen}

Note for windows users: Only [RTLD_NOLOAD] and [RTLD_NODELETE] are supported.
Passing no or any other flags to {!dlopen} will result in standard behaviour:
just LoadLibrary is called. If [RTLD_NOLOAD] is specified and the module is
not already loaded, a {!DL_error} with the string "library not loaded" is
thrown; there is however no test, if such a library exists at all (like under
linux).
*)
type flag = 
    RTLD_LAZY
  | RTLD_NOW
  | RTLD_GLOBAL
  | RTLD_LOCAL
  | RTLD_NODELETE
  | RTLD_NOLOAD
  | RTLD_DEEPBIND

val dlopen : ?filename:string -> flags:flag list -> library
(** Open a dynamic library.

Note for windows users: the filename must be encoded in UTF-8 *)

val dlclose : handle:library -> unit
(** Close a dynamic library. *)

val dlsym : ?handle:library -> symbol:string -> Ctypes_ptr.voidp
(** Look up a symbol in a dynamic library. *)

end = struct
#1 "dl.ml"
(*
 * Copyright (c) 2013 Jeremy Yallop.
 *
 * This file is distributed under the terms of the MIT License.
 * See the file LICENSE for details.
 *)

type library

type flag = 
    RTLD_LAZY
  | RTLD_NOW
  | RTLD_GLOBAL
  | RTLD_LOCAL
  | RTLD_NODELETE
  | RTLD_NOLOAD
  | RTLD_DEEPBIND

exception DL_error of string

(* void *dlopen(const char *filename, int flag); *)
external _dlopen : ?filename:string -> flags:int -> library option
  = "ctypes_dlopen"
    
(* void *dlsym(void *handle, const char *symbol); *)
external _dlsym : ?handle:library -> symbol:string -> nativeint option
  = "ctypes_dlsym"

(* int dlclose(void *handle); *)
external _dlclose : handle:library -> int
  = "ctypes_dlclose"

(* char *dlerror(void); *)
external _dlerror : unit -> string option
  = "ctypes_dlerror"

external resolve_flag : flag -> int
  = "ctypes_resolve_dl_flag"

let _report_dl_error noload =
  match _dlerror () with
    | Some error -> raise (DL_error (error))
    | None       ->
      if noload then
        raise (DL_error "library not loaded")
      else
        failwith "dl_error: expected error, but no error reported"

let crush_flags f : 'a list -> int = List.fold_left (fun i o -> i lor (f o)) 0

let dlopen ?filename ~flags =
  match _dlopen ?filename ~flags:(crush_flags resolve_flag flags) with
    | Some library -> library
    | None         -> _report_dl_error (List.mem RTLD_NOLOAD flags)

let dlclose ~handle =
  match _dlclose ~handle with
    | 0 -> ()
    | _ -> _report_dl_error false

let dlsym ?handle ~symbol =
  match _dlsym ?handle ~symbol with
    | Some symbol -> Ctypes_ptr.Raw.of_nativeint symbol
    | None        -> _report_dl_error false

end
module Ctypes_foreign_basis
= struct
#1 "ctypes_foreign_basis.ml"
(*
 * Copyright (c) 2013 Jeremy Yallop.
 *
 * This file is distributed under the terms of the MIT License.
 * See the file LICENSE for details.
 *)

module Make(Closure_properties : Ctypes_ffi.CLOSURE_PROPERTIES) =
struct
  open Dl
  open Ctypes

  module Ffi = Ctypes_ffi.Make(Closure_properties)

  exception CallToExpiredClosure = Ctypes_ffi_stubs.CallToExpiredClosure

  let funptr ?(abi=Libffi_abi.default_abi) ?name ?(check_errno=false)
      ?(runtime_lock=false) ?(thread_registration=false) fn =
    let open Ffi in
    let read = function_of_pointer
      ~abi ~check_errno ~release_runtime_lock:runtime_lock ?name fn
    and write = pointer_of_function fn
      ~abi ~acquire_runtime_lock:runtime_lock ~thread_registration in
    Ctypes_static.(view ~read ~write (static_funptr fn))

  let funptr_opt ?abi ?name ?check_errno ?runtime_lock ?thread_registration fn =
    Ctypes_std_views.nullable_funptr_view
      (funptr ?abi ?name ?check_errno ?runtime_lock ?thread_registration fn) fn

  let funptr_of_raw_ptr p = 
    Ctypes.funptr_of_raw_address (Ctypes_ptr.Raw.to_nativeint p)

  let ptr_of_raw_ptr p = 
    Ctypes.ptr_of_raw_address (Ctypes_ptr.Raw.to_nativeint p)

  let foreign_value ?from symbol t =
    from_voidp t (ptr_of_raw_ptr (dlsym ?handle:from ~symbol))

  let foreign ?(abi=Libffi_abi.default_abi) ?from ?(stub=false)
      ?(check_errno=false) ?(release_runtime_lock=false) symbol typ =
    try
      let coerce = Ctypes_coerce.coerce (static_funptr (void @-> returning void))
        (funptr ~abi ~name:symbol ~check_errno ~runtime_lock:release_runtime_lock typ) in
      coerce (funptr_of_raw_ptr (dlsym ?handle:from ~symbol))
    with
    | exn -> if stub then fun _ -> raise exn else raise exn
end

end
module Ctypes_gc_mutex
= struct
#1 "ctypes_gc_mutex.ml"
(*
 * Copyright (c) 2013 Jeremy Yallop.
 *
 * This file is distributed under the terms of the MIT License.
 * See the file LICENSE for details.
 *)

(* For internal use only, and really only for use with Closure_properties_base.
   A mutex for synchronizing between the GC (i.e. finalisers) and the single
   mutator thread.  Provides very few guarantees.  Since the program is
   single-threaded, there is no waiting; locking either succeeds or fails
   immediately.
*)

exception MutexError of string

type t = { mutable locked: bool }

let create () = { locked = false }

(* the only allocation below is exception raising *) 

let lock m =
  if m.locked then raise (MutexError "Locking locked mutex")
  else m.locked <- true

let try_lock m = 
  if m.locked then false
  else (m.locked <- true; true)

let unlock m = 
  if not m.locked then raise (MutexError "Unlocking unlocked mutex")
  else m.locked <- false

end
module Foreign : sig 
#1 "foreign.mli"
(*
 * Copyright (c) 2013 Jeremy Yallop.
 *
 * This file is distributed under the terms of the MIT License.
 * See the file LICENSE for details.
 *)

(** High-level bindings for C functions and values *)

val foreign :
  ?abi:Libffi_abi.abi ->
  ?from:Dl.library ->
  ?stub:bool -> 
  ?check_errno:bool ->
  ?release_runtime_lock:bool ->
  string ->
  ('a -> 'b) Ctypes.fn ->
  ('a -> 'b)
(** [foreign name typ] exposes the C function of type [typ] named by [name] as
    an OCaml value.

    The argument [?from], if supplied, is a library handle returned by
    {!Dl.dlopen}.

    The argument [?stub], if [true] (defaults to [false]), indicates that the
    function should not raise an exception if [name] is not found but return
    an OCaml value that raises an exception when called.

    The value [?check_errno], which defaults to [false], indicates whether
    {!Unix.Unix_error} should be raised if the C function modifies [errno].
    Please note that a function that succeeds is allowed to change errno. So
    use this option with caution.

    The value [?release_runtime_lock], which defaults to [false], indicates
    whether the OCaml runtime lock should be released during the call to the C
    function, allowing other threads to run.  If the runtime lock is released
    then the C function must not access OCaml heap objects, such as arguments
    passed using {!Ctypes.ocaml_string} and {!Ctypes.ocaml_bytes}, and must not
    call back into OCaml.

    @raise Dl.DL_error if [name] is not found in [?from] and [?stub] is
    [false]. *)

val foreign_value : ?from:Dl.library -> string -> 'a Ctypes.typ -> 'a Ctypes.ptr
(** [foreign_value name typ] exposes the C value of type [typ] named by [name]
    as an OCaml value.  The argument [?from], if supplied, is a library handle
    returned by {!Dl.dlopen}.  *)

val funptr :
  ?abi:Libffi_abi.abi ->
  ?name:string ->
  ?check_errno:bool ->
  ?runtime_lock:bool ->
  ?thread_registration:bool ->
  ('a -> 'b) Ctypes.fn ->
  ('a -> 'b) Ctypes.typ
(** Construct a function pointer type from a function type.

    The ctypes library, like C itself, distinguishes functions and function
    pointers.  Functions are not first class: it is not possible to use them
    as arguments or return values of calls, or store them in addressable
    memory.  Function pointers are first class, and so have none of these
    restrictions.

    The value [?check_errno], which defaults to [false], indicates whether
    {!Unix.Unix_error} should be raised if the C function modifies [errno].

    The value [?runtime_lock], which defaults to [false], indicates whether
    the OCaml runtime lock should be released during the call to the C
    function, allowing other threads to run.  If the runtime lock is released
    then the C function must not access OCaml heap objects, such as arguments
    passed using {!Ctypes.ocaml_string} and {!Ctypes.ocaml_bytes}, and must
    not call back into OCaml.  If the function pointer is used to call into
    OCaml from C then the [?runtime_lock] argument indicates whether the lock
    should be acquired and held during the call.

    @raise Dl.DL_error if [name] is not found in [?from] and [?stub] is
    [false]. *)

val funptr_opt :
  ?abi:Libffi_abi.abi ->
  ?name:string ->
  ?check_errno:bool ->
  ?runtime_lock:bool ->
  ?thread_registration:bool ->
  ('a -> 'b) Ctypes.fn ->
  ('a -> 'b) option Ctypes.typ
(** Construct a function pointer type from a function type.

    This behaves like {!funptr}, except that null pointers appear in OCaml as
    [None]. *)

exception CallToExpiredClosure
(** A closure passed to C was collected by the OCaml garbage collector before
    it was called. *)

end = struct
#1 "foreign.ml"
(*
 * Copyright (c) 2013 Jeremy Yallop.
 *
 * This file is distributed under the terms of the MIT License.
 * See the file LICENSE for details.
 *)

include Ctypes_foreign_basis.Make(Ctypes_closure_properties.Make(Ctypes_gc_mutex))

end
module Result
= struct
#1 "result.ml"
type ('a, 'b) result = Ok of 'a | Error of 'b

end
module Tsdl_consts
= struct
#1 "tsdl_consts.ml"
let sdl_init_timer = 1
let sdl_init_audio = 16
let sdl_init_video = 32
let sdl_init_joystick = 512
let sdl_init_haptic = 4096
let sdl_init_gamecontroller = 8192
let sdl_init_events = 16384
let sdl_init_everything = 29233
let sdl_init_noparachute = 1048576
let sdl_hint_framebuffer_acceleration = "SDL_FRAMEBUFFER_ACCELERATION"
let sdl_hint_idle_timer_disabled = "SDL_IOS_IDLE_TIMER_DISABLED"
let sdl_hint_orientations = "SDL_IOS_ORIENTATIONS"
let sdl_hint_render_driver = "SDL_RENDER_DRIVER"
let sdl_hint_render_opengl_shaders = "SDL_RENDER_OPENGL_SHADERS"
let sdl_hint_render_scale_quality = "SDL_RENDER_SCALE_QUALITY"
let sdl_hint_render_vsync = "SDL_RENDER_VSYNC"
let sdl_hint_default = 0
let sdl_hint_normal = 1
let sdl_hint_override = 2
let sdl_log_category_application = 0
let sdl_log_category_error = 1
let sdl_log_category_system = 3
let sdl_log_category_audio = 4
let sdl_log_category_video = 5
let sdl_log_category_render = 6
let sdl_log_category_input = 7
let sdl_log_category_custom = 19
let sdl_log_priority_verbose = 1
let sdl_log_priority_debug = 2
let sdl_log_priority_info = 3
let sdl_log_priority_warn = 4
let sdl_log_priority_error = 5
let sdl_log_priority_critical = 6
let sdl_blendmode_none = 0
let sdl_blendmode_blend = 1
let sdl_blendmode_add = 2
let sdl_blendmode_mod = 4
let sdl_pixelformat_unknown = 0x0l
let sdl_pixelformat_index1lsb = 0x11100100l
let sdl_pixelformat_index1msb = 0x11200100l
let sdl_pixelformat_index4lsb = 0x12100400l
let sdl_pixelformat_index4msb = 0x12200400l
let sdl_pixelformat_index8 = 0x13000801l
let sdl_pixelformat_rgb332 = 0x14110801l
let sdl_pixelformat_rgb444 = 0x15120C02l
let sdl_pixelformat_rgb555 = 0x15130F02l
let sdl_pixelformat_bgr555 = 0x15530F02l
let sdl_pixelformat_argb4444 = 0x15321002l
let sdl_pixelformat_rgba4444 = 0x15421002l
let sdl_pixelformat_abgr4444 = 0x15721002l
let sdl_pixelformat_bgra4444 = 0x15821002l
let sdl_pixelformat_argb1555 = 0x15331002l
let sdl_pixelformat_rgba5551 = 0x15441002l
let sdl_pixelformat_abgr1555 = 0x15731002l
let sdl_pixelformat_bgra5551 = 0x15841002l
let sdl_pixelformat_rgb565 = 0x15151002l
let sdl_pixelformat_bgr565 = 0x15551002l
let sdl_pixelformat_rgb24 = 0x17101803l
let sdl_pixelformat_bgr24 = 0x17401803l
let sdl_pixelformat_rgb888 = 0x16161804l
let sdl_pixelformat_rgbx8888 = 0x16261804l
let sdl_pixelformat_bgr888 = 0x16561804l
let sdl_pixelformat_bgrx8888 = 0x16661804l
let sdl_pixelformat_argb8888 = 0x16362004l
let sdl_pixelformat_rgba8888 = 0x16462004l
let sdl_pixelformat_abgr8888 = 0x16762004l
let sdl_pixelformat_bgra8888 = 0x16862004l
let sdl_pixelformat_argb2101010 = 0x16372004l
let sdl_pixelformat_yv12 = 0x32315659l
let sdl_pixelformat_iyuv = 0x56555949l
let sdl_pixelformat_yuy2 = 0x32595559l
let sdl_pixelformat_uyvy = 0x59565955l
let sdl_pixelformat_yvyu = 0x55595659l
let sdl_flip_none = 0
let sdl_flip_horizontal = 1
let sdl_flip_vertical = 2
let sdl_renderer_software = 1
let sdl_renderer_accelerated = 2
let sdl_renderer_presentvsync = 4
let sdl_renderer_targettexture = 8
let sdl_textureaccess_static = 0
let sdl_textureaccess_streaming = 1
let sdl_textureaccess_target = 2
let sdl_texturemodulate_none = 0
let sdl_texturemodulate_color = 1
let sdl_texturemodulate_alpha = 2
let sdl_window_fullscreen = 1
let sdl_window_fullscreen_desktop = 4097
let sdl_window_opengl = 2
let sdl_window_shown = 4
let sdl_window_hidden = 8
let sdl_window_borderless = 16
let sdl_window_resizable = 32
let sdl_window_minimized = 64
let sdl_window_maximized = 128
let sdl_window_input_grabbed = 256
let sdl_window_input_focus = 512
let sdl_window_mouse_focus = 1024
let sdl_window_foreign = 2048
let sdl_window_allow_highdpi = 8192
let sdl_windowpos_centered = 805240832
let sdl_windowpos_undefined = 536805376
let sdl_gl_context_debug_flag = 1
let sdl_gl_context_forward_compatible_flag = 2
let sdl_gl_context_robust_access_flag = 4
let sdl_gl_context_reset_isolation_flag = 8
let sdl_gl_context_profile_core = 1
let sdl_gl_context_profile_compatibility = 2
let sdl_gl_context_profile_es = 4
let sdl_gl_red_size = 0
let sdl_gl_green_size = 1
let sdl_gl_blue_size = 2
let sdl_gl_alpha_size = 3
let sdl_gl_buffer_size = 4
let sdl_gl_doublebuffer = 5
let sdl_gl_depth_size = 6
let sdl_gl_stencil_size = 7
let sdl_gl_accum_red_size = 8
let sdl_gl_accum_green_size = 9
let sdl_gl_accum_blue_size = 10
let sdl_gl_accum_alpha_size = 11
let sdl_gl_stereo = 12
let sdl_gl_multisamplebuffers = 13
let sdl_gl_multisamplesamples = 14
let sdl_gl_accelerated_visual = 15
let sdl_gl_context_major_version = 17
let sdl_gl_context_minor_version = 18
let sdl_gl_context_egl = 19
let sdl_gl_context_flags = 20
let sdl_gl_context_profile_mask = 21
let sdl_gl_share_with_current_context = 22
let sdl_gl_framebuffer_srgb_capable = 23
let sdl_messagebox_error = 16
let sdl_messagebox_warning = 32
let sdl_messagebox_information = 64
let sdl_messagebox_button_returnkey_default = 1
let sdl_messagebox_button_escapekey_default = 2
let sdl_messagebox_color_background = 0
let sdl_messagebox_color_text = 1
let sdl_messagebox_color_button_border = 2
let sdl_messagebox_color_button_background = 3
let sdl_messagebox_color_button_selected = 4
let sdl_messagebox_color_max = 5
let sdl_scancode_unknown = 0
let sdl_scancode_a = 4
let sdl_scancode_b = 5
let sdl_scancode_c = 6
let sdl_scancode_d = 7
let sdl_scancode_e = 8
let sdl_scancode_f = 9
let sdl_scancode_g = 10
let sdl_scancode_h = 11
let sdl_scancode_i = 12
let sdl_scancode_j = 13
let sdl_scancode_k = 14
let sdl_scancode_l = 15
let sdl_scancode_m = 16
let sdl_scancode_n = 17
let sdl_scancode_o = 18
let sdl_scancode_p = 19
let sdl_scancode_q = 20
let sdl_scancode_r = 21
let sdl_scancode_s = 22
let sdl_scancode_t = 23
let sdl_scancode_u = 24
let sdl_scancode_v = 25
let sdl_scancode_w = 26
let sdl_scancode_x = 27
let sdl_scancode_y = 28
let sdl_scancode_z = 29
let sdl_scancode_1 = 30
let sdl_scancode_2 = 31
let sdl_scancode_3 = 32
let sdl_scancode_4 = 33
let sdl_scancode_5 = 34
let sdl_scancode_6 = 35
let sdl_scancode_7 = 36
let sdl_scancode_8 = 37
let sdl_scancode_9 = 38
let sdl_scancode_0 = 39
let sdl_scancode_return = 40
let sdl_scancode_escape = 41
let sdl_scancode_backspace = 42
let sdl_scancode_tab = 43
let sdl_scancode_space = 44
let sdl_scancode_minus = 45
let sdl_scancode_equals = 46
let sdl_scancode_leftbracket = 47
let sdl_scancode_rightbracket = 48
let sdl_scancode_backslash = 49
let sdl_scancode_nonushash = 50
let sdl_scancode_semicolon = 51
let sdl_scancode_apostrophe = 52
let sdl_scancode_grave = 53
let sdl_scancode_comma = 54
let sdl_scancode_period = 55
let sdl_scancode_slash = 56
let sdl_scancode_capslock = 57
let sdl_scancode_f1 = 58
let sdl_scancode_f2 = 59
let sdl_scancode_f3 = 60
let sdl_scancode_f4 = 61
let sdl_scancode_f5 = 62
let sdl_scancode_f6 = 63
let sdl_scancode_f7 = 64
let sdl_scancode_f8 = 65
let sdl_scancode_f9 = 66
let sdl_scancode_f10 = 67
let sdl_scancode_f11 = 68
let sdl_scancode_f12 = 69
let sdl_scancode_printscreen = 70
let sdl_scancode_scrolllock = 71
let sdl_scancode_pause = 72
let sdl_scancode_insert = 73
let sdl_scancode_home = 74
let sdl_scancode_pageup = 75
let sdl_scancode_delete = 76
let sdl_scancode_end = 77
let sdl_scancode_pagedown = 78
let sdl_scancode_right = 79
let sdl_scancode_left = 80
let sdl_scancode_down = 81
let sdl_scancode_up = 82
let sdl_scancode_numlockclear = 83
let sdl_scancode_kp_divide = 84
let sdl_scancode_kp_multiply = 85
let sdl_scancode_kp_minus = 86
let sdl_scancode_kp_plus = 87
let sdl_scancode_kp_enter = 88
let sdl_scancode_kp_1 = 89
let sdl_scancode_kp_2 = 90
let sdl_scancode_kp_3 = 91
let sdl_scancode_kp_4 = 92
let sdl_scancode_kp_5 = 93
let sdl_scancode_kp_6 = 94
let sdl_scancode_kp_7 = 95
let sdl_scancode_kp_8 = 96
let sdl_scancode_kp_9 = 97
let sdl_scancode_kp_0 = 98
let sdl_scancode_kp_period = 99
let sdl_scancode_nonusbackslash = 100
let sdl_scancode_application = 101
let sdl_scancode_kp_equals = 103
let sdl_scancode_f13 = 104
let sdl_scancode_f14 = 105
let sdl_scancode_f15 = 106
let sdl_scancode_f16 = 107
let sdl_scancode_f17 = 108
let sdl_scancode_f18 = 109
let sdl_scancode_f19 = 110
let sdl_scancode_f20 = 111
let sdl_scancode_f21 = 112
let sdl_scancode_f22 = 113
let sdl_scancode_f23 = 114
let sdl_scancode_f24 = 115
let sdl_scancode_execute = 116
let sdl_scancode_help = 117
let sdl_scancode_menu = 118
let sdl_scancode_select = 119
let sdl_scancode_stop = 120
let sdl_scancode_again = 121
let sdl_scancode_undo = 122
let sdl_scancode_cut = 123
let sdl_scancode_copy = 124
let sdl_scancode_paste = 125
let sdl_scancode_find = 126
let sdl_scancode_mute = 127
let sdl_scancode_volumeup = 128
let sdl_scancode_volumedown = 129
let sdl_scancode_kp_comma = 133
let sdl_scancode_kp_equalsas400 = 134
let sdl_scancode_international1 = 135
let sdl_scancode_international2 = 136
let sdl_scancode_international3 = 137
let sdl_scancode_international4 = 138
let sdl_scancode_international5 = 139
let sdl_scancode_international6 = 140
let sdl_scancode_international7 = 141
let sdl_scancode_international8 = 142
let sdl_scancode_international9 = 143
let sdl_scancode_lang1 = 144
let sdl_scancode_lang2 = 145
let sdl_scancode_lang3 = 146
let sdl_scancode_lang4 = 147
let sdl_scancode_lang5 = 148
let sdl_scancode_lang6 = 149
let sdl_scancode_lang7 = 150
let sdl_scancode_lang8 = 151
let sdl_scancode_lang9 = 152
let sdl_scancode_alterase = 153
let sdl_scancode_sysreq = 154
let sdl_scancode_cancel = 155
let sdl_scancode_clear = 156
let sdl_scancode_prior = 157
let sdl_scancode_return2 = 158
let sdl_scancode_separator = 159
let sdl_scancode_out = 160
let sdl_scancode_oper = 161
let sdl_scancode_clearagain = 162
let sdl_scancode_crsel = 163
let sdl_scancode_exsel = 164
let sdl_scancode_kp_00 = 176
let sdl_scancode_kp_000 = 177
let sdl_scancode_thousandsseparator = 178
let sdl_scancode_decimalseparator = 179
let sdl_scancode_currencyunit = 180
let sdl_scancode_currencysubunit = 181
let sdl_scancode_kp_leftparen = 182
let sdl_scancode_kp_rightparen = 183
let sdl_scancode_kp_leftbrace = 184
let sdl_scancode_kp_rightbrace = 185
let sdl_scancode_kp_tab = 186
let sdl_scancode_kp_backspace = 187
let sdl_scancode_kp_a = 188
let sdl_scancode_kp_b = 189
let sdl_scancode_kp_c = 190
let sdl_scancode_kp_d = 191
let sdl_scancode_kp_e = 192
let sdl_scancode_kp_f = 193
let sdl_scancode_kp_xor = 194
let sdl_scancode_kp_power = 195
let sdl_scancode_kp_percent = 196
let sdl_scancode_kp_less = 197
let sdl_scancode_kp_greater = 198
let sdl_scancode_kp_ampersand = 199
let sdl_scancode_kp_dblampersand = 200
let sdl_scancode_kp_verticalbar = 201
let sdl_scancode_kp_dblverticalbar = 202
let sdl_scancode_kp_colon = 203
let sdl_scancode_kp_hash = 204
let sdl_scancode_kp_space = 205
let sdl_scancode_kp_at = 206
let sdl_scancode_kp_exclam = 207
let sdl_scancode_kp_memstore = 208
let sdl_scancode_kp_memrecall = 209
let sdl_scancode_kp_memclear = 210
let sdl_scancode_kp_memadd = 211
let sdl_scancode_kp_memsubtract = 212
let sdl_scancode_kp_memmultiply = 213
let sdl_scancode_kp_memdivide = 214
let sdl_scancode_kp_plusminus = 215
let sdl_scancode_kp_clear = 216
let sdl_scancode_kp_clearentry = 217
let sdl_scancode_kp_binary = 218
let sdl_scancode_kp_octal = 219
let sdl_scancode_kp_decimal = 220
let sdl_scancode_kp_hexadecimal = 221
let sdl_scancode_lctrl = 224
let sdl_scancode_lshift = 225
let sdl_scancode_lalt = 226
let sdl_scancode_lgui = 227
let sdl_scancode_rctrl = 228
let sdl_scancode_rshift = 229
let sdl_scancode_ralt = 230
let sdl_scancode_rgui = 231
let sdl_scancode_mode = 257
let sdl_scancode_audionext = 258
let sdl_scancode_audioprev = 259
let sdl_scancode_audiostop = 260
let sdl_scancode_audioplay = 261
let sdl_scancode_audiomute = 262
let sdl_scancode_mediaselect = 263
let sdl_scancode_www = 264
let sdl_scancode_mail = 265
let sdl_scancode_calculator = 266
let sdl_scancode_computer = 267
let sdl_scancode_ac_search = 268
let sdl_scancode_ac_home = 269
let sdl_scancode_ac_back = 270
let sdl_scancode_ac_forward = 271
let sdl_scancode_ac_stop = 272
let sdl_scancode_ac_refresh = 273
let sdl_scancode_ac_bookmarks = 274
let sdl_scancode_brightnessdown = 275
let sdl_scancode_brightnessup = 276
let sdl_scancode_displayswitch = 277
let sdl_scancode_kbdillumtoggle = 278
let sdl_scancode_kbdillumdown = 279
let sdl_scancode_kbdillumup = 280
let sdl_scancode_eject = 281
let sdl_scancode_sleep = 282
let sdl_scancode_app1 = 283
let sdl_scancode_app2 = 284
let sdl_num_scancodes = 512
let sdlk_scancode_mask = 0x40000000
let sdlk_unknown = 0x0
let sdlk_return = 0xD
let sdlk_escape = 0x1B
let sdlk_backspace = 0x8
let sdlk_tab = 0x9
let sdlk_space = 0x20
let sdlk_exclaim = 0x21
let sdlk_quotedbl = 0x22
let sdlk_hash = 0x23
let sdlk_percent = 0x25
let sdlk_dollar = 0x24
let sdlk_ampersand = 0x26
let sdlk_quote = 0x27
let sdlk_leftparen = 0x28
let sdlk_rightparen = 0x29
let sdlk_asterisk = 0x2A
let sdlk_plus = 0x2B
let sdlk_comma = 0x2C
let sdlk_minus = 0x2D
let sdlk_period = 0x2E
let sdlk_slash = 0x2F
let sdlk_0 = 0x30
let sdlk_1 = 0x31
let sdlk_2 = 0x32
let sdlk_3 = 0x33
let sdlk_4 = 0x34
let sdlk_5 = 0x35
let sdlk_6 = 0x36
let sdlk_7 = 0x37
let sdlk_8 = 0x38
let sdlk_9 = 0x39
let sdlk_colon = 0x3A
let sdlk_semicolon = 0x3B
let sdlk_less = 0x3C
let sdlk_equals = 0x3D
let sdlk_greater = 0x3E
let sdlk_question = 0x3F
let sdlk_at = 0x40
let sdlk_leftbracket = 0x5B
let sdlk_backslash = 0x5C
let sdlk_rightbracket = 0x5D
let sdlk_caret = 0x5E
let sdlk_underscore = 0x5F
let sdlk_backquote = 0x60
let sdlk_a = 0x61
let sdlk_b = 0x62
let sdlk_c = 0x63
let sdlk_d = 0x64
let sdlk_e = 0x65
let sdlk_f = 0x66
let sdlk_g = 0x67
let sdlk_h = 0x68
let sdlk_i = 0x69
let sdlk_j = 0x6A
let sdlk_k = 0x6B
let sdlk_l = 0x6C
let sdlk_m = 0x6D
let sdlk_n = 0x6E
let sdlk_o = 0x6F
let sdlk_p = 0x70
let sdlk_q = 0x71
let sdlk_r = 0x72
let sdlk_s = 0x73
let sdlk_t = 0x74
let sdlk_u = 0x75
let sdlk_v = 0x76
let sdlk_w = 0x77
let sdlk_x = 0x78
let sdlk_y = 0x79
let sdlk_z = 0x7A
let sdlk_capslock = 0x40000039
let sdlk_f1 = 0x4000003A
let sdlk_f2 = 0x4000003B
let sdlk_f3 = 0x4000003C
let sdlk_f4 = 0x4000003D
let sdlk_f5 = 0x4000003E
let sdlk_f6 = 0x4000003F
let sdlk_f7 = 0x40000040
let sdlk_f8 = 0x40000041
let sdlk_f9 = 0x40000042
let sdlk_f10 = 0x40000043
let sdlk_f11 = 0x40000044
let sdlk_f12 = 0x40000045
let sdlk_printscreen = 0x40000046
let sdlk_scrolllock = 0x40000047
let sdlk_pause = 0x40000048
let sdlk_insert = 0x40000049
let sdlk_home = 0x4000004A
let sdlk_pageup = 0x4000004B
let sdlk_delete = 0x7F
let sdlk_end = 0x4000004D
let sdlk_pagedown = 0x4000004E
let sdlk_right = 0x4000004F
let sdlk_left = 0x40000050
let sdlk_down = 0x40000051
let sdlk_up = 0x40000052
let sdlk_numlockclear = 0x40000053
let sdlk_kp_divide = 0x40000054
let sdlk_kp_multiply = 0x40000055
let sdlk_kp_minus = 0x40000056
let sdlk_kp_plus = 0x40000057
let sdlk_kp_enter = 0x40000058
let sdlk_kp_1 = 0x40000059
let sdlk_kp_2 = 0x4000005A
let sdlk_kp_3 = 0x4000005B
let sdlk_kp_4 = 0x4000005C
let sdlk_kp_5 = 0x4000005D
let sdlk_kp_6 = 0x4000005E
let sdlk_kp_7 = 0x4000005F
let sdlk_kp_8 = 0x40000060
let sdlk_kp_9 = 0x40000061
let sdlk_kp_0 = 0x40000062
let sdlk_kp_period = 0x40000063
let sdlk_application = 0x40000065
let sdlk_power = 0x40000066
let sdlk_kp_equals = 0x40000067
let sdlk_f13 = 0x40000068
let sdlk_f14 = 0x40000069
let sdlk_f15 = 0x4000006A
let sdlk_f16 = 0x4000006B
let sdlk_f17 = 0x4000006C
let sdlk_f18 = 0x4000006D
let sdlk_f19 = 0x4000006E
let sdlk_f20 = 0x4000006F
let sdlk_f21 = 0x40000070
let sdlk_f22 = 0x40000071
let sdlk_f23 = 0x40000072
let sdlk_f24 = 0x40000073
let sdlk_execute = 0x40000074
let sdlk_help = 0x40000075
let sdlk_menu = 0x40000076
let sdlk_select = 0x40000077
let sdlk_stop = 0x40000078
let sdlk_again = 0x40000079
let sdlk_undo = 0x4000007A
let sdlk_cut = 0x4000007B
let sdlk_copy = 0x4000007C
let sdlk_paste = 0x4000007D
let sdlk_find = 0x4000007E
let sdlk_mute = 0x4000007F
let sdlk_volumeup = 0x40000080
let sdlk_volumedown = 0x40000081
let sdlk_kp_comma = 0x40000085
let sdlk_kp_equalsas400 = 0x40000086
let sdlk_alterase = 0x40000099
let sdlk_sysreq = 0x4000009A
let sdlk_cancel = 0x4000009B
let sdlk_clear = 0x4000009C
let sdlk_prior = 0x4000009D
let sdlk_return2 = 0x4000009E
let sdlk_separator = 0x4000009F
let sdlk_out = 0x400000A0
let sdlk_oper = 0x400000A1
let sdlk_clearagain = 0x400000A2
let sdlk_crsel = 0x400000A3
let sdlk_exsel = 0x400000A4
let sdlk_kp_00 = 0x400000B0
let sdlk_kp_000 = 0x400000B1
let sdlk_thousandsseparator = 0x400000B2
let sdlk_decimalseparator = 0x400000B3
let sdlk_currencyunit = 0x400000B4
let sdlk_currencysubunit = 0x400000B5
let sdlk_kp_leftparen = 0x400000B6
let sdlk_kp_rightparen = 0x400000B7
let sdlk_kp_leftbrace = 0x400000B8
let sdlk_kp_rightbrace = 0x400000B9
let sdlk_kp_tab = 0x400000BA
let sdlk_kp_backspace = 0x400000BB
let sdlk_kp_a = 0x400000BC
let sdlk_kp_b = 0x400000BD
let sdlk_kp_c = 0x400000BE
let sdlk_kp_d = 0x400000BF
let sdlk_kp_e = 0x400000C0
let sdlk_kp_f = 0x400000C1
let sdlk_kp_xor = 0x400000C2
let sdlk_kp_power = 0x400000C3
let sdlk_kp_percent = 0x400000C4
let sdlk_kp_less = 0x400000C5
let sdlk_kp_greater = 0x400000C6
let sdlk_kp_ampersand = 0x400000C7
let sdlk_kp_dblampersand = 0x400000C8
let sdlk_kp_verticalbar = 0x400000C9
let sdlk_kp_dblverticalbar = 0x400000CA
let sdlk_kp_colon = 0x400000CB
let sdlk_kp_hash = 0x400000CC
let sdlk_kp_space = 0x400000CD
let sdlk_kp_at = 0x400000CE
let sdlk_kp_exclam = 0x400000CF
let sdlk_kp_memstore = 0x400000D0
let sdlk_kp_memrecall = 0x400000D1
let sdlk_kp_memclear = 0x400000D2
let sdlk_kp_memadd = 0x400000D3
let sdlk_kp_memsubtract = 0x400000D4
let sdlk_kp_memmultiply = 0x400000D5
let sdlk_kp_memdivide = 0x400000D6
let sdlk_kp_plusminus = 0x400000D7
let sdlk_kp_clear = 0x400000D8
let sdlk_kp_clearentry = 0x400000D9
let sdlk_kp_binary = 0x400000DA
let sdlk_kp_octal = 0x400000DB
let sdlk_kp_decimal = 0x400000DC
let sdlk_kp_hexadecimal = 0x400000DD
let sdlk_lctrl = 0x400000E0
let sdlk_lshift = 0x400000E1
let sdlk_lalt = 0x400000E2
let sdlk_lgui = 0x400000E3
let sdlk_rctrl = 0x400000E4
let sdlk_rshift = 0x400000E5
let sdlk_ralt = 0x400000E6
let sdlk_rgui = 0x400000E7
let sdlk_mode = 0x40000101
let sdlk_audionext = 0x40000102
let sdlk_audioprev = 0x40000103
let sdlk_audiostop = 0x40000104
let sdlk_audioplay = 0x40000105
let sdlk_audiomute = 0x40000106
let sdlk_mediaselect = 0x40000107
let sdlk_www = 0x40000108
let sdlk_mail = 0x40000109
let sdlk_calculator = 0x4000010A
let sdlk_computer = 0x4000010B
let sdlk_ac_search = 0x4000010C
let sdlk_ac_home = 0x4000010D
let sdlk_ac_back = 0x4000010E
let sdlk_ac_forward = 0x4000010F
let sdlk_ac_stop = 0x40000110
let sdlk_ac_refresh = 0x40000111
let sdlk_ac_bookmarks = 0x40000112
let sdlk_brightnessdown = 0x40000113
let sdlk_brightnessup = 0x40000114
let sdlk_displayswitch = 0x40000115
let sdlk_kbdillumtoggle = 0x40000116
let sdlk_kbdillumdown = 0x40000117
let sdlk_kbdillumup = 0x40000118
let sdlk_eject = 0x40000119
let sdlk_sleep = 0x4000011A
let kmod_none = 0x0
let kmod_lshift = 0x1
let kmod_rshift = 0x2
let kmod_lctrl = 0x40
let kmod_rctrl = 0x80
let kmod_lalt = 0x100
let kmod_ralt = 0x200
let kmod_lgui = 0x400
let kmod_rgui = 0x800
let kmod_num = 0x1000
let kmod_caps = 0x2000
let kmod_mode = 0x4000
let kmod_reserved = 0x8000
let kmod_ctrl = 0xC0
let kmod_shift = 0x3
let kmod_alt = 0x300
let kmod_gui = 0xC00
let sdl_system_cursor_arrow = 0
let sdl_system_cursor_ibeam = 1
let sdl_system_cursor_wait = 2
let sdl_system_cursor_crosshair = 3
let sdl_system_cursor_waitarrow = 4
let sdl_system_cursor_sizenwse = 5
let sdl_system_cursor_sizenesw = 6
let sdl_system_cursor_sizewe = 7
let sdl_system_cursor_sizens = 8
let sdl_system_cursor_sizeall = 9
let sdl_system_cursor_no = 10
let sdl_system_cursor_hand = 11
let sdl_button_left = 1
let sdl_button_middle = 2
let sdl_button_right = 3
let sdl_button_x1 = 4
let sdl_button_x2 = 5
let sdl_button_lmask = 1
let sdl_button_mmask = 2
let sdl_button_rmask = 4
let sdl_button_x1mask = 8
let sdl_button_x2mask = 16
let sdl_touch_mouseid = 0xFFFFFFFFl
let sdl_hat_centered = 0
let sdl_hat_up = 1
let sdl_hat_right = 2
let sdl_hat_down = 4
let sdl_hat_left = 8
let sdl_hat_rightup = 3
let sdl_hat_rightdown = 6
let sdl_hat_leftup = 9
let sdl_hat_leftdown = 12
let sdl_controller_bindtype_none = 0
let sdl_controller_bindtype_button = 1
let sdl_controller_bindtype_axis = 2
let sdl_controller_bindtype_hat = 3
let sdl_controller_axis_invalid = -1
let sdl_controller_axis_leftx = 0
let sdl_controller_axis_lefty = 1
let sdl_controller_axis_rightx = 2
let sdl_controller_axis_righty = 3
let sdl_controller_axis_triggerleft = 4
let sdl_controller_axis_triggerright = 5
let sdl_controller_axis_max = 6
let sdl_controller_button_invalid = -1
let sdl_controller_button_a = 0
let sdl_controller_button_b = 1
let sdl_controller_button_x = 2
let sdl_controller_button_y = 3
let sdl_controller_button_back = 4
let sdl_controller_button_guide = 5
let sdl_controller_button_start = 6
let sdl_controller_button_leftstick = 7
let sdl_controller_button_rightstick = 8
let sdl_controller_button_leftshoulder = 9
let sdl_controller_button_rightshoulder = 10
let sdl_controller_button_dpad_up = 11
let sdl_controller_button_dpad_down = 12
let sdl_controller_button_dpad_left = 13
let sdl_controller_button_dpad_right = 14
let sdl_controller_button_max = 15
let sdl_query = -1
let sdl_disable = 0
let sdl_enable = 1
let sdl_pressed = 1
let sdl_released = 0
let sdl_firstevent = 0
let sdl_quit = 256
let sdl_app_terminating = 257
let sdl_app_lowmemory = 258
let sdl_app_willenterbackground = 259
let sdl_app_didenterbackground = 260
let sdl_app_willenterforeground = 261
let sdl_app_didenterforeground = 262
let sdl_windowevent = 512
let sdl_syswmevent = 513
let sdl_keydown = 768
let sdl_keyup = 769
let sdl_textediting = 770
let sdl_textinput = 771
let sdl_mousemotion = 1024
let sdl_mousebuttondown = 1025
let sdl_mousebuttonup = 1026
let sdl_mousewheel = 1027
let sdl_joyaxismotion = 1536
let sdl_joyballmotion = 1537
let sdl_joyhatmotion = 1538
let sdl_joybuttondown = 1539
let sdl_joybuttonup = 1540
let sdl_joydeviceadded = 1541
let sdl_joydeviceremoved = 1542
let sdl_controlleraxismotion = 1616
let sdl_controllerbuttondown = 1617
let sdl_controllerbuttonup = 1618
let sdl_controllerdeviceadded = 1619
let sdl_controllerdeviceremoved = 1620
let sdl_controllerdeviceremapped = 1621
let sdl_fingerdown = 1792
let sdl_fingerup = 1793
let sdl_fingermotion = 1794
let sdl_dollargesture = 2048
let sdl_dollarrecord = 2049
let sdl_multigesture = 2050
let sdl_clipboardupdate = 2304
let sdl_dropfile = 4096
let sdl_userevent = 32768
let sdl_lastevent = 65535
let tsdl_sdl_event_size = 56
let sdl_texteditingevent_text_size = 32
let sdl_textinputevent_text_size = 32
let sdl_windowevent_shown = 1
let sdl_windowevent_hidden = 2
let sdl_windowevent_exposed = 3
let sdl_windowevent_moved = 4
let sdl_windowevent_resized = 5
let sdl_windowevent_size_changed = 6
let sdl_windowevent_minimized = 7
let sdl_windowevent_maximized = 8
let sdl_windowevent_restored = 9
let sdl_windowevent_enter = 10
let sdl_windowevent_leave = 11
let sdl_windowevent_focus_gained = 12
let sdl_windowevent_focus_lost = 13
let sdl_windowevent_close = 14
let sdl_haptic_constant = 1
let sdl_haptic_sine = 2
let sdl_haptic_leftright = 4
let sdl_haptic_triangle = 8
let sdl_haptic_sawtoothup = 16
let sdl_haptic_sawtoothdown = 32
let sdl_haptic_ramp = 64
let sdl_haptic_spring = 128
let sdl_haptic_damper = 256
let sdl_haptic_inertia = 512
let sdl_haptic_friction = 1024
let sdl_haptic_custom = 2048
let sdl_haptic_gain = 4096
let sdl_haptic_autocenter = 8192
let sdl_haptic_status = 16384
let sdl_haptic_pause = 32768
let sdl_haptic_polar = 0
let sdl_haptic_cartesian = 1
let sdl_haptic_spherical = 2
let sdl_audio_stopped = 0
let sdl_audio_playing = 1
let sdl_audio_paused = 2
let audio_s8 = 32776
let audio_u8 = 8
let audio_s16lsb = 32784
let audio_s16msb = 36880
let audio_s16sys = 32784
let audio_s16 = 32784
let audio_s16lsb = 32784
let audio_u16lsb = 16
let audio_u16msb = 4112
let audio_u16sys = 16
let audio_u16 = 16
let audio_u16lsb = 16
let audio_s32lsb = 32800
let audio_s32msb = 36896
let audio_s32sys = 32800
let audio_s32 = 32800
let audio_s32lsb = 32800
let audio_f32lsb = 33056
let audio_f32msb = 37152
let audio_f32sys = 33056
let audio_f32 = 33056
let sdl_audio_allow_frequency_change = 1
let sdl_audio_allow_format_change = 2
let sdl_audio_allow_channels_change = 4
let sdl_audio_allow_any_change = 7
let sdl_powerstate_unknown = 0
let sdl_powerstate_on_battery = 1
let sdl_powerstate_no_battery = 2
let sdl_powerstate_charging = 3
let sdl_powerstate_charged = 4

end
module Tsdl : sig 
#1 "tsdl.mli"
(*---------------------------------------------------------------------------
   Copyright (c) 2013 Daniel C. Bünzli. All rights reserved.
   Distributed under the ISC license, see terms at the end of the file.
   tsdl v0.9.1
  ---------------------------------------------------------------------------*)

(** SDL thin bindings.

    Consult the {{!conventions}binding conventions}, the
    {{!Sdl.coverage}binding coverage} and {{!examples}examples} of
    use.  Given the thinness of the binding most functions are
    documented by linking directly to SDL's own documentation.

    Open the module to use it, this defines only the module [Sdl] in
    your scope.

    {b Note.} The module initialization code calls
    {{:http://wiki.libsdl.org/SDL_SetMainReady}SDL_SetMainReady}.

    {b References}
    {ul
    {- {{:http://wiki.libsdl.org/APIByCategory}SDL API}}}

    {e Release v0.9.1 — SDL 2.0.3 —
    {{:http://erratique.ch/software/tsdl }homepage}} *)

(** {1:sdl SDL} *)

(** SDL bindings.

    {ul
    {- {!Sdl.basics}
    {ul
    {- {{!section:Sdl.init}Initialization and shutdown}}
    {- {{!Sdl.hints}Hints}}
    {- {{!Sdl.errors}Errors}}
    {- {{!Sdl.log}Log}}
    {- {{!Sdl.version}Version}}
    }}
    {- {!Sdl.fileabstraction}
    {ul
    {- {{!Sdl.io}IO abstraction}}
    {- {{!Sdl.fspaths}Filesystem paths}}
    }}
    {- {!Sdl.video}
    {ul
    {- {{!Sdl.colors}Colors}}
    {- {{!Sdl.points}Points}}
    {- {{!Sdl.rectangles}Rectangles}}
    {- {{!Sdl.palettes}Pallettes}}
    {- {{!Sdl.pixel_formats}Pixel formats}}
    {- {{!Sdl.surfaces}Surfaces}}
    {- {{!Sdl.renderers}Renderers}}
    {- {{!Sdl.textures}Textures}}
    {- {{!Sdl.videodrivers}Video drivers}}
    {- {{!Sdl.displays}Displays}}
    {- {{!Sdl.windows}Windows}}
    {- {{!Sdl.opengl}OpenGL contexts}}
    {- {{!Sdl.screensaver}Screen saver}}
    {- {{!Sdl.messageboxes}Message boxes}}
    }}
    {- {!Sdl.input}
    {ul
    {- {{!Sdl.keyboard}Keyboard}}
    {- {{!Sdl.mouse}Mouse}}
    {- {{!Sdl.touch}Touch and gestures}}
    {- {{!Sdl.joystick}Joystick}}
    {- {{!Sdl.gamecontroller}Game controller}}
    {- {{!Sdl.events}Events}}
    }}
    {- {{!Sdl.forcefeedback}Force feedback}}
    {- {{!Sdl.audio}Audio}
    {ul
    {- {{!Sdl.audiodrivers}Audio drivers}}
    {- {{!Sdl.audiodevices}Audio devices}}
    }}
    {- {{!Sdl.timer}Timer}}
    {- {!Sdl.platform}}
    {- {{!Sdl.power}Power}}
    {- {!Sdl.coverage}}}
*)
module Sdl : sig

(** {1:types Integer types, bigarrays and results} *)

type uint8 = int
type int16 = int
type uint16 = int
type uint32 = int32
type uint64 = int64
type ('a, 'b) bigarray = ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t
(** The type for bigarrays.*)

type 'a result = ('a, [ `Msg of string ]) Result.result
(** The type for function results. In the error case,
    the string is what {!Sdl.get_error} returned. *)

(** {1:basics Basics} *)

(** {2:init {{:http://wiki.libsdl.org/CategoryInit}
             Initialization and shutdown}} *)

module Init : sig
  type t

  val ( + ) : t -> t -> t
  (** [f + f'] combines flags [f] and [f']. *)

  val test : t -> t -> bool
  (** [test flags mask] is [true] if any of the flags in [mask] is
      set in [flags]. *)

  val eq : t -> t -> bool
  (** [eq f f'] is [true] if the flags are equal. *)

  val timer : t
  val audio : t
  val video : t
  val joystick : t
  val haptic : t
  val gamecontroller : t
  val events : t
  val everything : t
  val noparachute : t
end
(** Subsystem flags. *)

val init : Init.t -> unit result
(** {{:http://wiki.libsdl.org/SDL_Init}SDL_Init} *)

val init_sub_system : Init.t -> unit result
(** {{:http://wiki.libsdl.org/SDL_InitSubSystem}SDL_InitSubSystem} *)

val quit : unit -> unit
(** {{:http://wiki.libsdl.org/SDL_Quit}SDL_Quit} *)

val quit_sub_system : Init.t -> unit
(** {{:http://wiki.libsdl.org/SDL_QuitSubSystem}SDL_QuitSubSystem} *)

val was_init : Init.t option -> Init.t
(** {{:http://wiki.libsdl.org/SDL_WasInit}SDL_WasInit} *)

(** {2:hints {{:http://wiki.libsdl.org/CategoryHints}Hints}} *)

module Hint : sig

  (** {1:hint Hints} *)

  type t = string

  val framebuffer_acceleration : t
  (** {{:http://wiki.libsdl.org/SDL_HINT_FRAMEBUFFER_ACCELERATION}
       SDL_HINT_FRAMEBUFFER_ACCELERATION} *)

  val idle_timer_disabled : t
  (** {{:http://wiki.libsdl.org/SDL_HINT_IDLE_TIMER_DISABLED}
       SDL_HINT_IDLE_TIMER_DISABLED} *)

  val orientations : t
  (** {{:http://wiki.libsdl.org/SDL_HINT_ORIENTATIONS}
      SDL_HINT_ORIENTATIONS} *)

  val render_driver : t
  (** {{:http://wiki.libsdl.org/SDL_HINT_RENDER_DRIVER}
      SDL_HINT_RENDER_DRIVER} *)

  val render_opengl_shaders : t
  (** {{:http://wiki.libsdl.org/SDL_HINT_RENDER_OPENGL_SHADERS}
      SDL_HINT_RENDER_OPENGL_SHADERS} *)

  val render_scale_quality : t
  (** {{:http://wiki.libsdl.org/SDL_HINT_RENDER_SCALE_QUALITY}
      SDL_HINT_RENDER_SCALE_QUALITY} *)

  val render_vsync : t
  (** {{:http://wiki.libsdl.org/SDL_HINT_RENDER_VSYNC}
      SDL_HINT_RENDER_VSYNC} *)

  (** {1:priority Priority} *)

  type priority
  (** {{:http://wiki.libsdl.org/SDL_HintPriority}SDL_HintPriority} *)

  val default : priority
  val normal : priority
  val override : priority
end

val clear_hints : unit -> unit
(** {{:http://wiki.libsdl.org/SDL_ClearHints}SDL_ClearHints} *)

val get_hint : Hint.t -> string option
(** {{:http://wiki.libsdl.org/SDL_GetHint}SDL_GetHint} *)

val set_hint : Hint.t -> string -> bool
(** {{:http://wiki.libsdl.org/SDL_SetHint}SDL_SetHint} *)

val set_hint_with_priority : Hint.t -> string -> Hint.priority -> bool
(** {{:http://wiki.libsdl.org/SDL_SetHintWithPriority}
    SDL_SetHintWithPriority} *)

(** {2:errors {{:http://wiki.libsdl.org/CategoryError}Errors}} *)

val clear_error : unit -> unit
(** {{:http://wiki.libsdl.org/SDL_ClearError}SDL_ClearError} *)

val get_error : unit -> string
(** {{:http://wiki.libsdl.org/SDL_GetError}SDL_GetError} *)

val set_error : ('b, Format.formatter, unit) format -> 'b
(** {{:http://wiki.libsdl.org/SDL_SetError}SDL_SetError} *)

(** {2:log {{:http://wiki.libsdl.org/CategoryLog}Log}} *)

module Log : sig

  (** {1:category Category} *)

  type category = int
  (** {{:http://wiki.libsdl.org/SDL_LOG_CATEGORY}SDL_LOG_CATEGORY} *)

  val category_application : category
  val category_error : category
  val category_system : category
  val category_audio : category
  val category_video : category
  val category_render : category
  val category_input : category

  (** {1:priority Priority} *)

  type priority
  val priority_compare : priority -> priority -> int
  val priority_verbose : priority
  val priority_debug : priority
  val priority_info : priority
  val priority_warn : priority
  val priority_error : priority
  val priority_critical : priority
end

val log : ('b, Format.formatter, unit) format -> 'b
(** {{:http://wiki.libsdl.org/SDL_Log}SDL_Log} *)

val log_critical : Log.category -> ('b, Format.formatter, unit) format -> 'b
(** {{:http://wiki.libsdl.org/SDL_LogCritical}SDL_LogCritical} *)

val log_debug : Log.category -> ('b, Format.formatter, unit) format -> 'b
(** {{:http://wiki.libsdl.org/SDL_LogDebug}SDL_LogDebug} *)

val log_error : Log.category -> ('b, Format.formatter, unit) format -> 'b
(** {{:http://wiki.libsdl.org/SDL_LogError}SDL_LogError} *)

val log_get_priority : Log.category -> Log.priority
(** {{:http://wiki.libsdl.org/SDL_LogGetPriority}SDL_LogGetPriority} *)

val log_info : Log.category -> ('b, Format.formatter, unit) format -> 'b
(** {{:http://wiki.libsdl.org/SDL_LogInfo}SDL_LogInfo} *)

val log_message : Log.category -> Log.priority ->
  ('b, Format.formatter, unit) format -> 'b
(** {{:http://wiki.libsdl.org/SDL_LogMessage}SDL_LogMessage} *)

val log_reset_priorities : unit -> unit
(** {{:http://wiki.libsdl.org/SDL_LogResetPriorities}SDL_LogResetPriorities} *)

val log_set_all_priority : Log.priority -> unit
(** {{:http://wiki.libsdl.org/SDL_LogSetAllPriority}SDL_LogSetAllPriority} *)

val log_set_priority : Log.category -> Log.priority -> unit
(** {{:http://wiki.libsdl.org/SDL_LogSetPriority}SDL_LogSetPriority} *)

val log_verbose : Log.category -> ('b, Format.formatter, unit) format -> 'b
(** {{:http://wiki.libsdl.org/SDL_LogVerbose}SDL_LogVerbose} *)

val log_warn : Log.category -> ('b, Format.formatter, unit) format -> 'b
(** {{:http://wiki.libsdl.org/SDL_LogWarn}SDL_LogWarn} *)

(** {2:version {{:http://wiki.libsdl.org/CategoryVersion}Version}} *)

val get_version : unit -> (int * int * int)
(** {{:http://wiki.libsdl.org/SDL_GetVersion}SDL_GetVersion} *)

val get_revision : unit -> string
(** {{:http://wiki.libsdl.org/SDL_GetRevision}SDL_GetRevision} *)

val get_revision_number : unit -> int
(** {{:http://wiki.libsdl.org/SDL_GetRevisionNumber}SDL_GetRevisionNumber} *)

(** {1:fileabstraction Files and IO abstraction} *)

(** {2:io {{:https://wiki.libsdl.org/CategoryIO}IO abstraction}} *)

type rw_ops
(** {{:https://wiki.libsdl.org/SDL_RWops}SDL_RWops} *)

val rw_from_file : string -> string -> rw_ops result
(** {{:https://wiki.libsdl.org/SDL_RWFromFile}SDL_RWFromFile} *)

val rw_close : rw_ops -> unit result
(** {{:https://wiki.libsdl.org/SDL_RWclose}SDL_RWclose} *)

(**/**)
val unsafe_rw_ops_of_ptr : nativeint -> rw_ops
val unsafe_ptr_of_rw_ops : rw_ops -> nativeint
(**/**)

(** {1:fspaths {{:https://wiki.libsdl.org/CategoryFilesystem}Filesystem
    Paths}} *)

val get_base_path : unit -> string result
(** {{:https://wiki.libsdl.org/SDL_GetBasePath}SDL_GetBasePath} *)

val get_pref_path : org:string -> app:string -> string result
(** {{:https://wiki.libsdl.org/SDL_GetPrefPath}SDL_GetPrefPath} *)

(** {1:video Video} *)

type window
(** SDL_Window *)

(**/**)
val unsafe_window_of_ptr : nativeint -> window
val unsafe_ptr_of_window : window -> nativeint
(**/**)

(** {2:colors Colors} *)

type color
(** {{:http://wiki.libsdl.org/SDL_Rect}SDL_Color} *)

module Color : sig
  val create : r:uint8 -> g:uint8 -> b:uint8 -> a:uint8 -> color
  val r : color -> uint8
  val g : color -> uint8
  val b : color -> uint8
  val a : color -> uint8
  val set_r : color -> uint8 -> unit
  val set_g : color -> uint8 -> unit
  val set_b : color -> uint8 -> unit
  val set_a : color -> uint8 -> unit
end

(** {2:points Points} *)

type point
(** {{:http://wiki.libsdl.org/SDL_Point}SDL_Point} *)

module Point : sig
  val create : x:int -> y:int -> point
  val x : point -> int
  val y : point -> int
  val set_x : point -> int -> unit
  val set_y : point -> int -> unit
end

(** {2:rectangles
    {{:http://wiki.libsdl.org/CategoryRect}Rectangles}} *)

type rect
(** {{:http://wiki.libsdl.org/SDL_Rect}SDL_Rect} *)

module Rect : sig
  val create : x:int -> y:int -> w:int -> h:int -> rect
  val x : rect -> int
  val y : rect -> int
  val w : rect -> int
  val h : rect -> int
  val set_x : rect -> int -> unit
  val set_y : rect -> int -> unit
  val set_w : rect -> int -> unit
  val set_h : rect -> int -> unit
end

val enclose_points : ?clip:rect -> point list -> rect option
(** {{:http://wiki.libsdl.org/SDL_EnclosePoints}SDL_EnclosePoints}.
    Returns [None] if all the points were outside
    the clipping rectangle (if provided). *)

val enclose_points_ba : ?clip:rect -> (int32, Bigarray.int32_elt) bigarray ->
  rect option
(** See {!enclose_points}. Each consecutive pair in the array defines a
    point.
    @raise Invalid_argument if the length of the array is not
    a multiple of 2. *)

val has_intersection : rect -> rect -> bool
(** {{:http://wiki.libsdl.org/SDL_HasIntersection}SDL_HasIntersection} *)

val intersect_rect : rect -> rect -> rect option
(** {{:http://wiki.libsdl.org/SDL_IntersectRect}SDL_IntersectRect} *)

val intersect_rect_and_line : rect -> int -> int -> int -> int ->
  ((int * int) * (int * int)) option
(** {{:http://wiki.libsdl.org/SDL_IntersectRectAndLine}
    SDL_IntersectRectAndLine}. Returns the clipped segment if it
    intersects. *)

val rect_empty : rect -> bool
(** {{:http://wiki.libsdl.org/SDL_RectEmpty}SDL_RectEmpty} *)

val rect_equals : rect -> rect -> bool
(** {{:http://wiki.libsdl.org/SDL_RectEquals}SDL_RectEquals} *)

val union_rect : rect -> rect -> rect
(** {{:http://wiki.libsdl.org/SDL_UnionRect}SDL_UnionRect} *)

(** {2:palettes {{:http://wiki.libsdl.org/CategoryPixels}Palettes}} *)

type palette
(** {{:https://wiki.libsdl.org/SDL_Palette}SDL_Palette} *)

val alloc_palette : int -> palette result
(** {{:http://wiki.libsdl.org/SDL_AllocPalette}SDL_AllocPalette} *)

val free_palette : palette -> unit
(** {{:http://wiki.libsdl.org/SDL_FreePalette}SDL_FreePalette} *)

val get_palette_ncolors : palette -> int
(** [get_palette_ncolors p] is the field [ncolors] of [p]. *)

val get_palette_colors : palette -> color list
(** [get_palette_colors p] is a copy of the contents of the field [colors]
    of [s]. *)

val get_palette_colors_ba : palette ->
  (int, Bigarray.int8_unsigned_elt) bigarray
(** [get_palette_colors_ba p] is a copy of the contents of the field [colors]
    of [p]. *)

val set_palette_colors : palette -> color list -> fst:int ->
  unit result
(** {{:http://wiki.libsdl.org/SDL_SetPaletteColors}SDL_SetPaletteColors} *)

val set_palette_colors_ba : palette ->
  (int, Bigarray.int8_unsigned_elt) bigarray -> fst:int -> unit result
(** See {!set_palette_colors}. Each consecutive quadruplet defines a
    color. The data is copied.
    @raise Invalid_argument if the length of the array is not
    a multiple of 4. *)

(**/**)
val unsafe_palette_of_ptr : nativeint -> palette
val unsafe_ptr_of_palette : palette -> nativeint
(**/**)

(** {2:pixel_formats {{:http://wiki.libsdl.org/CategoryPixels}Pixels
    formats}} *)

type gamma_ramp = (int, Bigarray.int16_unsigned_elt) bigarray
(** The type for gamma ramps, 256 [uint16] values. *)

val calculate_gamma_ramp : float -> gamma_ramp
(** {{:http://wiki.libsdl.org/SDL_CalculateGammaRamp}SDL_CalculateGammaRamp} *)

module Blend : sig
  type mode
  (** {{:https://wiki.libsdl.org/SDL_BlendMode}SDL_BlendMode} *)

  val mode_none : mode
  val mode_blend : mode
  val mode_add : mode
  val mode_mod : mode
end

module Pixel : sig
  type format_enum
  (** {{:https://wiki.libsdl.org/SDL_PixelFormatEnum}SDL_PixelFormatEnum}. *)

  val eq : format_enum -> format_enum -> bool
  val to_uint32 : format_enum -> uint32
  val format_unknown : format_enum
  val format_index1lsb : format_enum
  val format_index1msb : format_enum
  val format_index4lsb : format_enum
  val format_index4msb : format_enum
  val format_index8 : format_enum
  val format_rgb332 : format_enum
  val format_rgb444 : format_enum
  val format_rgb555 : format_enum
  val format_bgr555 : format_enum
  val format_argb4444 : format_enum
  val format_rgba4444 : format_enum
  val format_abgr4444 : format_enum
  val format_bgra4444 : format_enum
  val format_argb1555 : format_enum
  val format_rgba5551 : format_enum
  val format_abgr1555 : format_enum
  val format_bgra5551 : format_enum
  val format_rgb565 : format_enum
  val format_bgr565 : format_enum
  val format_rgb24 : format_enum
  val format_bgr24 : format_enum
  val format_rgb888 : format_enum
  val format_rgbx8888 : format_enum
  val format_bgr888 : format_enum
  val format_bgrx8888 : format_enum
  val format_argb8888 : format_enum
  val format_rgba8888 : format_enum
  val format_abgr8888 : format_enum
  val format_bgra8888 : format_enum
  val format_argb2101010 : format_enum
  val format_yv12 : format_enum
  val format_iyuv : format_enum
  val format_yuy2 : format_enum
  val format_uyvy : format_enum
  val format_yvyu : format_enum
end

type pixel_format
(** {{:https://wiki.libsdl.org/SDL_PixelFormat}SDL_PixelFormat} *)

val alloc_format : Pixel.format_enum -> pixel_format result
(** {{:http://wiki.libsdl.org/SDL_AllocFormat}SDL_AllocFormat} *)

val free_format : pixel_format -> unit
(** {{:http://wiki.libsdl.org/SDL_FreeFormat}SDL_FreeFormat} *)

val get_pixel_format_name : Pixel.format_enum -> string
(** {{:http://wiki.libsdl.org/SDL_GetPixelFormatName}SDL_GetPixelFormatName} *)

val get_pixel_format_format : pixel_format -> Pixel.format_enum
(** [get_pixel_format_bytes_pp pf] is the field [format] of [pf]. *)

val get_pixel_format_bits_pp : pixel_format -> int
(** [get_pixel_format_bytes_pp pf] is the field [BitsPerPixel] of [pf]. *)

val get_pixel_format_bytes_pp : pixel_format -> int
(** [get_pixel_format_bytes_pp pf] is the field [BytesPerPixel] of [pf]. *)

val get_rgb : pixel_format -> uint32 -> (uint8 * uint8 * uint8)
(** {{:http://wiki.libsdl.org/SDL_GetRGB}SDL_GetRGB} *)

val get_rgba : pixel_format -> uint32 -> (uint8 * uint8 * uint8 * uint8)
(** {{:http://wiki.libsdl.org/SDL_GetRGBA}SDL_GetRGBA} *)

val map_rgb : pixel_format -> uint8 -> uint8 -> uint8 -> uint32
(** {{:http://wiki.libsdl.org/SDL_MapRGB}SDL_MapRGB} *)

val map_rgba : pixel_format -> uint8 -> uint8 -> uint8 -> uint8 -> uint32
(** {{:http://wiki.libsdl.org/SDL_MapRGBA}SDL_MapRGBA} *)

val masks_to_pixel_format_enum :
  int -> uint32 -> uint32 -> uint32 -> uint32 -> Pixel.format_enum
(** {{:http://wiki.libsdl.org/SDL_MasksToPixelFormatEnum}
    SDL_MasksToPixelFormatEnum} *)

val pixel_format_enum_to_masks :
  Pixel.format_enum -> (int * uint32 * uint32 * uint32 * uint32) result
(** {{:http://wiki.libsdl.org/SDL_PixelFormatEnumToMasks}
    SDL_PixelFormatEnumToMasks} *)

val set_pixel_format_palette : pixel_format -> palette -> unit result
(** {{:http://wiki.libsdl.org/SDL_SetPixelFormatPalette}
    SDL_SetPixelFormatPalette}.

    {b Note} If you allocated the palette with {!alloc_palette} you
    can {!free_palette} after. *)

(**/**)
val unsafe_pixel_format_of_ptr : nativeint -> pixel_format
val unsafe_ptr_of_pixel_format : pixel_format -> nativeint
(**/**)

(** {2:surfaces
    {{:http://wiki.libsdl.org/CategorySurface}Surface}} *)

type surface
(** {{:https://wiki.libsdl.org/SDL_Surface}SDL_Surface} *)

val blit_scaled : src:surface -> rect -> dst:surface -> rect option ->
  unit result
(** {{:http://wiki.libsdl.org/SDL_BlitScaled}SDL_BlitScaled} *)

val blit_surface : src:surface -> rect option -> dst:surface -> rect option ->
  unit result
(** {{:http://wiki.libsdl.org/SDL_BlitSurface}SDL_BlitSurface} *)

val convert_pixels : w:int -> h:int -> src:Pixel.format_enum ->
  ('a, 'b) bigarray -> int -> dst:Pixel.format_enum ->
  ('c, 'd) bigarray -> int -> unit result
(** {{:http://wiki.libsdl.org/SDL_ConvertPixels}SDL_ConvertPixels}

    {b Note} Pitches are given in bigarray elements {b not} in bytes.

    {b Warning.} Unsafe, make sure your parameters don't result
    in invalid access to memory. *)

val convert_surface : surface -> pixel_format -> surface result
(** {{:http://wiki.libsdl.org/SDL_ConvertSurface}SDL_ConvertSurface} *)

val convert_surface_format : surface -> Pixel.format_enum -> surface result
(** {{:http://wiki.libsdl.org/SDL_ConvertSurfaceFormat}
    SDL_ConvertSurfaceFormat} *)

val create_rgb_surface : w:int -> h:int -> depth:int -> uint32 -> uint32 ->
  uint32 -> uint32 -> surface result
(** {{:http://wiki.libsdl.org/SDL_CreateRGBSurface}SDL_CreateRGBSurface} *)

val create_rgb_surface_from : ('a, 'b) bigarray -> w:int -> h:int ->
  depth:int -> pitch:int -> uint32 -> uint32 -> uint32 -> uint32 ->
  surface result
(** {{:http://wiki.libsdl.org/SDL_CreateRGBSurfaceFrom}
    SDL_CreateRGBSurfaceFrom}

    {b Note} The pitch is given in bigarray elements {b not} in
    bytes.

    {b Warning} Unsafe, make sure your parameters don't result
    in invalid access to memory. The bigarray data is not copied,
    it must remain valid until {!free_surface} is called on the
    surface. *)

val fill_rect : surface -> rect option -> uint32 -> unit result
(** {{:http://wiki.libsdl.org/SDL_FillRect}SDL_FillRect} *)

val fill_rects : surface -> rect list -> uint32 -> unit result
(** {{:http://wiki.libsdl.org/SDL_FillRects}SDL_FillRects} *)

val fill_rects_ba : surface -> (int32, Bigarray.int32_elt) bigarray ->
  uint32 -> unit result
(** See {!fill_rects}. Each consecutive quadruplet defines a
    rectangle.
    @raise Invalid_argument if the length of the array is not
    a multiple of 4. *)

val free_surface : surface -> unit
(** {{:http://wiki.libsdl.org/SDL_FreeSurface}SDL_FreeSurface} *)

val get_clip_rect : surface -> rect
(** {{:http://wiki.libsdl.org/SDL_GetClipRect}SDL_GetClipRect} *)

val get_color_key : surface -> uint32 result
(** {{:http://wiki.libsdl.org/SDL_GetColorKey}SDL_GetColorKey} *)

val get_surface_alpha_mod : surface -> uint8 result
(** {{:http://wiki.libsdl.org/SDL_GetSurfaceAlphaMod}SDL_GetSurfaceAlphaMod} *)

val get_surface_blend_mode : surface -> Blend.mode result
(** {{:http://wiki.libsdl.org/SDL_GetSurfaceBlendMode}
    SDL_GetSurfaceBlendMode} *)

val get_surface_color_mod : surface -> (int * int * int) result
(** {{:http://wiki.libsdl.org/SDL_GetSurfaceColorMod}
    SDL_GetSurfaceColorMod} *)

val get_surface_format_enum : surface -> Pixel.format_enum
(** [get_surface_format_neum s] is the pixel format enum of the
    field [format] of [s]. *)

val get_surface_pitch : surface -> int
(** [get_surface_pitch s] is the field [pitch] of [s]. *)

val get_surface_pixels : surface -> ('a, 'b) Bigarray.kind -> ('a, 'b) bigarray
(** [get_surface_pixels s kind] is the field [pixels] of [s] viewed as
    a [kind] bigarray. Note that you must lock the surface before
    accessing this.

    {b Warning.} The bigarray memory becomes invalid
    once the surface is unlocked or freed.

    @raise Invalid_argument If [kind] can't align with the surface pitch. *)

val get_surface_size : surface -> int * int
(** [get_surface_size s] is the fields [w] and [h] of [s]. *)

val load_bmp : string -> surface result
(** {{:http://wiki.libsdl.org/SDL_LoadBMP}SDL_LoadBMP} *)

val load_bmp_rw : rw_ops -> close:bool -> surface result
(** {{:http://wiki.libsdl.org/SDL_LoadBMP_RW}SDL_LoadBMP_RW} *)

val lock_surface : surface -> unit result
(** {{:http://wiki.libsdl.org/SDL_LockSurface}SDL_LockSurface} *)

val lower_blit : src:surface -> rect -> dst:surface -> rect ->
  unit result
(** {{:http://wiki.libsdl.org/SDL_LowerBlit}SDL_LowerBlit} *)

val lower_blit_scaled : src:surface -> rect -> dst:surface -> rect ->
  unit result
(** {{:http://wiki.libsdl.org/SDL_LowerBlitScaled}SDL_LowerBlitScaled} *)

val save_bmp : surface -> string -> unit result
(** {{:http://wiki.libsdl.org/SDL_SaveBMP}SDL_SaveBMP} *)

val save_bmp_rw : surface -> rw_ops -> close:bool -> unit result
(** {{:http://wiki.libsdl.org/SDL_SaveBMP_RW}SDL_SaveBMP_RW} *)

val set_clip_rect : surface -> rect -> bool
(** {{:http://wiki.libsdl.org/SDL_SetClipRect}SDL_SetClipRect} *)

val set_color_key : surface -> bool -> uint32 -> unit result
(** {{:http://wiki.libsdl.org/SDL_SetColorKey}SDL_SetColorKey} *)

val set_surface_alpha_mod : surface -> uint8 -> unit result
(** {{:http://wiki.libsdl.org/SDL_SetSurfaceAlphaMod}SDL_SetSurfaceAlphaMod} *)

val set_surface_blend_mode : surface -> Blend.mode -> unit result
(** {{:http://wiki.libsdl.org/SDL_SetSurfaceBlendMode}
    SDL_SetSurfaceBlendMode} *)

val set_surface_color_mod : surface -> uint8 -> uint8 -> uint8 -> unit result
(** {{:http://wiki.libsdl.org/SDL_SetSurfaceColorMod}SDL_SetSurfaceColorMod} *)

val set_surface_palette : surface -> palette -> unit result
(** {{:http://wiki.libsdl.org/SDL_SetSurfacePalette}SDL_SetSurfacePalette}

    {b Note} If you allocated the palette with {!alloc_palette} you
    can {!free_palette} after. *)

val set_surface_rle : surface -> bool -> unit result
(** {{:http://wiki.libsdl.org/SDL_SetSurfaceRLE}SDL_SetSurfaceRLE} *)

val unlock_surface : surface -> unit
(** {{:http://wiki.libsdl.org/SDL_UnlockSurface}SDL_UnlockSurface} *)

(**/**)
val unsafe_surface_of_ptr : nativeint -> surface
val unsafe_ptr_of_surface : surface -> nativeint
(**/**)

(** {2:renderers {{:http://wiki.libsdl.org/CategoryRender}Renderers}} *)

type flip
(** {{:https://wiki.libsdl.org/SDL_RendererFlip}SDL_RendererFlip} *)

module Flip : sig
  val ( + ) : flip  -> flip -> flip
  (** [f + f'] combines flips [f] and [f']. *)

  val none : flip
  val horizontal : flip
  val vertical : flip
end

type texture
(** SDL_Texture *)

(**/**)
val unsafe_texture_of_ptr : nativeint -> texture
val unsafe_ptr_of_texture : texture -> nativeint
(**/**)

type renderer

(**/**)
val unsafe_renderer_of_ptr : nativeint -> renderer
val unsafe_ptr_of_renderer : renderer -> nativeint
(**/**)

(** SDL_Renderer *)

module Renderer : sig
  type flags
  (** {{:https://wiki.libsdl.org/SDL_RendererFlags}SDL_RendererFlags} *)

  val ( + ) : flags -> flags -> flags
  (** [f + f'] combines flags [f] and [f']. *)

  val test : flags -> flags -> bool
  (** [test flags mask] is [true] if any of the flags in [mask] is
        set in [flags]. *)

  val eq : flags -> flags -> bool
  (** [eq f f'] is [true] if the flags are equal. *)

  val software : flags
  val accelerated : flags
  val presentvsync : flags
  val targettexture : flags
end

type renderer_info =
  { ri_name : string;
    ri_flags : Renderer.flags;
    ri_texture_formats : Pixel.format_enum list;
    ri_max_texture_width : int;
    ri_max_texture_height : int; }
(** {{:https://wiki.libsdl.org/SDL_RendererInfo}SDL_RendererInfo} *)

val create_renderer : ?index:int -> ?flags:Renderer.flags-> window ->
  renderer result
(** {{:http://wiki.libsdl.org/SDL_CreateRenderer}SDL_CreateRenderer} *)

val create_software_renderer : surface -> renderer result
(** {{:http://wiki.libsdl.org/SDL_CreateSoftwareRenderer}
    SDL_CreateSoftwareRenderer} *)

val destroy_renderer : renderer -> unit
(** {{:http://wiki.libsdl.org/SDL_DestroyRenderer}SDL_DestroyRenderer} *)

val get_num_render_drivers : unit -> int result
(** {{:http://wiki.libsdl.org/SDL_GetNumRenderDrivers}
    SDL_GetNumRenderDrivers} *)

val get_render_draw_blend_mode : renderer -> Blend.mode result
(** {{:http://wiki.libsdl.org/SDL_GetRenderDrawBlendMode}
    SDL_GetRenderDrawBlendMode} *)

val get_render_draw_color : renderer -> (uint8 * uint8 * uint8 * uint8) result
(** {{:http://wiki.libsdl.org/SDL_GetRenderDrawColor}
    SDL_GetRenderDrawColor} *)

val get_render_driver_info : int -> renderer_info result
(** {{:http://wiki.libsdl.org/SDL_GetRenderDriverInfo}
    SDL_GetRenderDriverInfo} *)

val get_render_target : renderer -> texture option
(** {{:http://wiki.libsdl.org/SDL_GetRenderTarget}SDL_GetRenderTarget} *)

val get_renderer : window -> renderer result
(** {{:http://wiki.libsdl.org/SDL_GetRenderer}SDL_GetRenderer} *)

val get_renderer_info : renderer -> renderer_info result
(** {{:http://wiki.libsdl.org/SDL_GetRendererInfo}SDL_GetRendererInfo} *)

val get_renderer_output_size : renderer -> (int * int) result
(** {{:http://wiki.libsdl.org/SDL_GetRendererOutputSize}
    SDL_GetRendererOutputSize} *)

val render_clear : renderer -> unit result
(** {{:http://wiki.libsdl.org/SDL_RenderClear}SDL_RenderClear} *)

val render_copy : ?src:rect -> ?dst:rect -> renderer -> texture ->
  unit result
(** {{:http://wiki.libsdl.org/SDL_RenderCopy}SDL_RenderCopy} *)

val render_copy_ex : ?src:rect -> ?dst:rect ->renderer -> texture ->
  float -> point option -> flip -> unit result
(** {{:http://wiki.libsdl.org/SDL_RenderCopyEx}SDL_RenderCopyEx} *)

val render_draw_line : renderer -> int -> int -> int -> int ->
  unit result
(** {{:http://wiki.libsdl.org/SDL_RenderDrawLine}SDL_RenderDrawLine} *)

val render_draw_lines : renderer -> point list -> unit result
(** {{:http://wiki.libsdl.org/SDL_RenderDrawLines}SDL_RenderDrawLines} *)

val render_draw_lines_ba : renderer -> (int32, Bigarray.int32_elt) bigarray ->
  unit result
(** See {!render_draw_lines}. Each consecutive pair in the array
    defines a point.

    @raise Invalid_argument if the length of the array is not a
    multiple of 2. *)

val render_draw_point : renderer -> int -> int -> unit result
(** {{:http://wiki.libsdl.org/SDL_RenderDrawPoint}SDL_RenderDrawPoint} *)

val render_draw_points : renderer -> point list -> unit result
(** {{:http://wiki.libsdl.org/SDL_RenderDrawPoints}SDL_RenderDrawPoints} *)

val render_draw_points_ba : renderer -> (int32, Bigarray.int32_elt) bigarray ->
  unit result
(** See {!render_draw_points}. Each consecutive pair in the array
    defines a point.

    @raise Invalid_argument if the length of the array is not a
    multiple of 2. *)

val render_draw_rect : renderer -> rect option -> unit result
(** {{:http://wiki.libsdl.org/SDL_RenderDrawRect}SDL_RenderDrawRect} *)

val render_draw_rects : renderer -> rect list -> unit result
(** {{:http://wiki.libsdl.org/SDL_RenderDrawRects}SDL_RenderDrawRects} *)

val render_draw_rects_ba : renderer -> (int32, Bigarray.int32_elt) bigarray ->
  unit result
(** See {!render_draw_rects}. Each consecutive quadruple in the array
    defines a rectangle.

    @raise Invalid_argument if the length of the array is not a
    multiple of 4. *)

val render_fill_rect : renderer -> rect option -> unit result
(** {{:http://wiki.libsdl.org/SDL_RenderFillRect}SDL_RenderFillRect} *)

val render_fill_rects : renderer -> rect list -> unit result
(** {{:http://wiki.libsdl.org/SDL_RenderDrawRects}SDL_RenderDrawRects} *)

val render_fill_rects_ba : renderer -> (int32, Bigarray.int32_elt) bigarray ->
  unit result
(** See {!render_draw_rects}. Each consecutive quadruple in the array
    defines a rectangle.

    @raise Invalid_argument if the length of the array is not a
    multiple of 4. *)

val render_get_clip_rect : renderer -> rect
(** {{:http://wiki.libsdl.org/SDL_RenderGetClipRect}SDL_RenderGetClipRect} *)

val render_get_logical_size : renderer -> int * int
(** {{:http://wiki.libsdl.org/SDL_RenderGetLogicalSize}
    SDL_RenderGetLogicalSize} *)

val render_get_scale : renderer -> float * float
(** {{:http://wiki.libsdl.org/SDL_RenderGetScale}SDL_RenderGetScale} *)

val render_get_viewport : renderer -> rect
(** {{:http://wiki.libsdl.org/SDL_RenderGetViewport}SDL_RenderGetViewport} *)

val render_present : renderer -> unit
(** {{:http://wiki.libsdl.org/SDL_RenderPresent}SDL_RenderPresent} *)

val render_read_pixels : renderer -> rect option -> Pixel.format_enum option ->
  ('a, 'b) bigarray -> int -> unit result
(** {{:http://wiki.libsdl.org/SDL_RenderReadPixels}SDL_RenderReadPixels} *)

val render_set_clip_rect : renderer -> rect option -> unit result
(** {{:http://wiki.libsdl.org/SDL_RenderSetClipRect}SDL_RenderSetClipRect} *)

val render_set_logical_size : renderer -> int -> int -> unit result
(** {{:http://wiki.libsdl.org/SDL_RenderSetLogicalSize}
    SDL_RenderSetLogicalSize} *)

val render_set_scale : renderer -> float -> float -> unit result
(** {{:http://wiki.libsdl.org/SDL_RenderSetScale}SDL_RenderSetScale} *)

val render_set_viewport : renderer -> rect option -> unit result
(** {{:http://wiki.libsdl.org/SDL_RenderSetViewport}SDL_RenderSetViewport} *)

val render_target_supported : renderer -> bool
(** {{:http://wiki.libsdl.org/SDL_RenderTargetSupported}
    SDL_RenderTargetSupported} *)

val set_render_draw_blend_mode : renderer -> Blend.mode -> unit result
(** {{:http://wiki.libsdl.org/SDL_SetRenderDrawBlendMode}
    SDL_SetRenderDrawBlendMode} *)

val set_render_draw_color : renderer -> uint8 -> uint8 -> uint8 -> uint8 ->
  unit result
(** {{:http://wiki.libsdl.org/SDL_SetRenderDrawColor}SDL_SetRenderDrawColor} *)

val set_render_target : renderer -> texture option -> unit result
(** {{:http://wiki.libsdl.org/SDL_SetRenderTarget}SDL_SetRenderTarget} *)

(** {2:textures {{:http://wiki.libsdl.org/CategoryRender}Textures}} *)

module Texture : sig
  type access
  (** {{:https://wiki.libsdl.org/SDL_TextureAccess}SDL_TextureAccess} *)
  val access_static : access
  val access_streaming : access
  val access_target : access

  type modulate
  (** {{:https://wiki.libsdl.org/SDL_TextureModulate}SDL_TextureModulate} *)
  val modulate_none : modulate
  val modulate_color : modulate
  val modulate_alpha : modulate
end

val create_texture : renderer -> Pixel.format_enum -> Texture.access ->
  w:int -> h:int -> texture result
(** {{:http://wiki.libsdl.org/SDL_CreateTexture}SDL_CreateTexture} *)

val create_texture_from_surface : renderer -> surface -> texture result
(** {{:http://wiki.libsdl.org/SDL_CreateTextureFromSurface}
    SDL_CreateTextureFromSurface} *)

val destroy_texture : texture -> unit
(** {{:http://wiki.libsdl.org/SDL_DestroyTexture}SDL_DestroyTexture} *)

val get_texture_alpha_mod : texture -> uint8 result
(** {{:http://wiki.libsdl.org/SDL_GetTextureAlphaMod}SDL_GetTextureAlphaMod} *)

val get_texture_blend_mode : texture -> Blend.mode result
(** {{:http://wiki.libsdl.org/SDL_GetTextureBlendMode}
    SDL_GetTextureBlendMode} *)

val get_texture_color_mod : texture -> (uint8 * uint8 * uint8) result
(** {{:http://wiki.libsdl.org/SDL_GetTextureColorMod}SDL_GetTextureColorMod}. *)

val lock_texture : texture -> rect option -> ('a, 'b) Bigarray.kind ->
  (('a, 'b) bigarray * int) result
(** {{:http://wiki.libsdl.org/SDL_LockTexture}SDL_LockTexture}

    {b Note.} The returned pitch is in bigarray element, {b not} in bytes.

    @raise Invalid_argument If [kind] can't align with the texture pitch. *)

val query_texture : texture ->
  (Pixel.format_enum * Texture.access * (int * int)) result
(** {{:http://wiki.libsdl.org/SDL_QueryTexture}SDL_QueryTexture} *)

val set_texture_alpha_mod : texture -> uint8 -> unit result
(** {{:http://wiki.libsdl.org/SDL_SetTextureAlphaMod}
    SDL_SetTextureAlphaMod} *)

val set_texture_blend_mode : texture -> Blend.mode -> unit result
(** {{:http://wiki.libsdl.org/SDL_SetTextureBlendMode}
    SDL_SetTextureBlendMode} *)

val set_texture_color_mod : texture -> uint8 -> uint8 -> uint8 -> unit result
(** {{:http://wiki.libsdl.org/SDL_SetTextureColorMod}
    SDL_SetTextureColorMod} *)

val unlock_texture : texture -> unit
(** {{:http://wiki.libsdl.org/SDL_UnlockTexture}SDL_UnlockTexture} *)

val update_texture : texture -> rect option -> ('a, 'b) bigarray -> int ->
  unit result
(** {{:http://wiki.libsdl.org/SDL_UpdateTexture}SDL_UpdateTexture}

    {b Note} The pitch is given in bigarray elements {b not} in
    bytes. *)

val update_yuv_texture : texture -> rect option ->
  y:(int, Bigarray.int8_unsigned_elt) bigarray -> int ->
  u:(int, Bigarray.int8_unsigned_elt) bigarray -> int ->
  v:(int, Bigarray.int8_unsigned_elt) bigarray -> int -> unit result
(** {{:http://wiki.libsdl.org/SDL_UpdateYUVTexture}SDL_UpdateYUVTexture} *)

(** {2:videodrivers {{:http://wiki.libsdl.org/CategoryVideo}Video drivers}} *)

val get_current_video_driver : unit -> string option
(** {{:http://wiki.libsdl.org/SDL_GetCurrentVideoDriver}
    SDL_GetCurrentVideoDriver} *)

val get_num_video_drivers : unit -> int result
(** {{:http://wiki.libsdl.org/SDL_GetNumVideoDrivers}SDL_GetNumVideoDrivers} *)

val get_video_driver : int -> string result
(** {{:http://wiki.libsdl.org/SDL_GetVideoDriver}SDL_GetVideoDriver} *)

val video_init : string option -> unit result
(** {{:http://wiki.libsdl.org/SDL_VideoInit}SDL_VideoInit} *)

val video_quit : unit -> unit
(** {{:http://wiki.libsdl.org/SDL_VideoQuit}SDL_VideoQuit} *)

(** {2:displays {{:http://wiki.libsdl.org/CategoryVideo}Displays}} *)

type driverdata
(** {b Note.} Nothing can be done with that. *)

type display_mode =
  { dm_format : Pixel.format_enum;
    dm_w : int;
    dm_h : int;
    dm_refresh_rate : int option;
    dm_driverdata : driverdata option }
(** {{:http://wiki.libsdl.org/SDL_DisplayMode}SDL_DisplayMode} *)

val get_closest_display_mode : int -> display_mode -> display_mode option
(** {{:http://wiki.libsdl.org/SDL_GetClosestDisplayMode}
    SDL_GetClosestDisplayMode} *)

val get_current_display_mode : int -> display_mode result
(** {{:http://wiki.libsdl.org/SDL_GetCurrentDisplayMode}
    SDL_GetCurrentDisplayMode} *)

val get_desktop_display_mode : int -> display_mode result
(** {{:http://wiki.libsdl.org/SDL_GetDesktopDisplayMode}
    SDL_GetDesktopDisplayMode} *)

val get_display_bounds : int -> rect result
(** {{:http://wiki.libsdl.org/SDL_GetDisplayBounds}SDL_GetDisplayBounds} *)

val get_display_mode : int -> int -> display_mode result
(** {{:http://wiki.libsdl.org/SDL_GetDisplayMode}SDL_GetDisplayMode} *)

val get_display_name : int -> string result
(** {{:http://wiki.libsdl.org/SDL_GetDisplayName}SDL_GetDisplayName} *)

val get_num_display_modes : int -> int result
(** {{:http://wiki.libsdl.org/SDL_GetNumDisplayModes}SDL_GetNumDisplayModes} *)


val get_num_video_displays : unit -> int result
(** {{:http://wiki.libsdl.org/SDL_GetNumVideoDisplays}
    SDL_GetNumVideoDisplays} *)

(** {2:windows {{:http://wiki.libsdl.org/CategoryVideo}Windows}} *)

module Window : sig

  (** {1:position Position} *)

  val pos_undefined : int
  val pos_centered : int

  (** {1:position Flags} *)

  type flags
  (** {{:http://wiki.libsdl.org/SDL_WindowFlags}SDL_WindowFlags} *)

  val ( + ) : flags -> flags -> flags
  (** [f + f'] combines flags [f] and [f']. *)

  val test : flags -> flags -> bool
  (** [test flags mask] is [true] if any of the flags in [mask] is
      set in [flags]. *)

  val eq : flags -> flags -> bool
  (** [eq f f'] is [true] if the flags are equal. *)

  val windowed : flags
  (** Equal to [0]. The flag doesn't exist in SDL, it's for using with
      {!set_window_fullscreen}. *)
  val fullscreen : flags
  val fullscreen_desktop : flags
  val opengl : flags
  val shown : flags
  val hidden : flags
  val borderless : flags
  val resizable : flags
  val minimized : flags
  val maximized : flags
  val input_grabbed : flags
  val input_focus : flags
  val mouse_focus : flags
  val foreign : flags
  val allow_highdpi : flags
end

val create_window : string -> ?x:int -> ?y:int -> w:int -> h:int ->
  Window.flags -> window result
(** {{:http://wiki.libsdl.org/SDL_CreateWindow}SDL_CreateWindow}

    [x] and [y] default to {!Window.pos_undefined}. *)

val create_window_and_renderer : w:int -> h:int -> Window.flags ->
  (window * renderer) result
(** {{:http://wiki.libsdl.org/SDL_CreateWindowAndRenderer}
    SDL_CreateWindowAndRenderer} *)

val destroy_window : window -> unit
(** {{:http://wiki.libsdl.org/SDL_DestroyWindow}SDL_DestroyWindow} *)

val get_window_brightness : window -> float
(** {{:http://wiki.libsdl.org/SDL_GetWindowBrightness}
    SDL_GetWindowBrightness} *)

val get_window_display_index : window -> int result
(** {{:http://wiki.libsdl.org/SDL_GetWindowDisplay}SDL_GetWindowDisplayIndex} *)

val get_window_display_mode : window -> display_mode result
(** {{:http://wiki.libsdl.org/SDL_GetWindowDisplayMode}
    SDL_GetWindowDisplayMode} *)

val get_window_flags : window -> Window.flags
(** {{:http://wiki.libsdl.org/SDL_GetWindowFlags}SDL_GetWindowFlags} *)

val get_window_from_id : int -> window result
(** {{:http://wiki.libsdl.org/SDL_GetWindowFromID}SDL_GetWindowFromID} *)

val get_window_gamma_ramp : window ->
  (gamma_ramp * gamma_ramp * gamma_ramp) result
(** {{:http://wiki.libsdl.org/SDL_GetWindowGammaRamp}
    SDL_GetWindowGammaRamp} *)

val get_window_grab : window -> bool
(** {{:http://wiki.libsdl.org/SDL_GetWindowGrab}SDL_GetWindowGrab} *)

val get_window_id : window -> int
(** {{:http://wiki.libsdl.org/SDL_GetWindowID}SDL_GetWindowID} *)

val get_window_maximum_size : window -> int * int
(** {{:http://wiki.libsdl.org/SDL_GetWindowMaximumSize}
    SDL_GetWindowMaximumSize} *)

val get_window_minimum_size : window -> int * int
(** {{:http://wiki.libsdl.org/SDL_GetWindowMinimumSize}
    SDL_GetWindowMinimumSize} *)

val get_window_pixel_format : window -> Pixel.format_enum
(** {{:http://wiki.libsdl.org/SDL_GetWindowPixelFormat}
    SDL_GetWindowPixelFormat} *)

val get_window_position : window -> int * int
(** {{:http://wiki.libsdl.org/SDL_GetWindowPosition}SDL_GetWindowPosition} *)

val get_window_size : window -> int * int
(** {{:http://wiki.libsdl.org/SDL_GetWindowSize}SDL_GetWindowSize} *)

val get_window_surface : window -> surface result
(** {{:http://wiki.libsdl.org/SDL_GetWindowSurface}SDL_GetWindowSurface}.

    {b Note}. According to SDL's documentation the surface
    is freed when the window is destroyed. *)

val get_window_title : window -> string
(** {{:http://wiki.libsdl.org/SDL_GetWindowTitle}SDL_GetWindowTitle} *)

val hide_window : window -> unit
(** {{:http://wiki.libsdl.org/SDL_HideWindow}SDL_HideWindow} *)

val maximize_window : window -> unit
(** {{:http://wiki.libsdl.org/SDL_MaximizeWindow}SDL_MaximizeWindow} *)

val minimize_window : window -> unit
(** {{:http://wiki.libsdl.org/SDL_MinimizeWindow}SDL_MinimizeWindow} *)

val raise_window : window -> unit
(** {{:http://wiki.libsdl.org/SDL_RaiseWindow}SDL_RaiseWindow} *)

val restore_window : window -> unit
(** {{:http://wiki.libsdl.org/SDL_RestoreWindow}SDL_RestoreWindow} *)

val set_window_bordered : window -> bool -> unit
(** {{:http://wiki.libsdl.org/SDL_SetWindowBordered}SDL_SetWindowBordered} *)

val set_window_brightness : window -> float -> unit result
(** {{:http://wiki.libsdl.org/SDL_SetWindowBrightness}
    SDL_SetWindowBrightness} *)

val set_window_display_mode : window -> display_mode -> unit result
(** {{:http://wiki.libsdl.org/SDL_SetWindowDisplayMode}
    SDL_SetWindowDisplayMode} *)

val set_window_fullscreen : window -> Window.flags -> unit result
(** {{:http://wiki.libsdl.org/SDL_SetWindowFullscreen}
    SDL_SetWindowFullscreen} *)

val set_window_gamma_ramp : window -> gamma_ramp -> gamma_ramp ->
  gamma_ramp -> unit result
(** {{:http://wiki.libsdl.org/SDL_SetWindowGammaRamp}SDL_SetWindowGammaRamp} *)

val set_window_grab : window -> bool -> unit
(** {{:http://wiki.libsdl.org/SDL_SetWindowGrab}SDL_SetWindowGrab} *)

val set_window_icon : window -> surface -> unit
(** {{:http://wiki.libsdl.org/SDL_SetWindowIcon}SDL_SetWindowIcon} *)

val set_window_maximum_size : window -> w:int -> h:int -> unit
(** {{:http://wiki.libsdl.org/SDL_SetWindowMaximumSize}
    SDL_SetWindowMaximumSize} *)

val set_window_minimum_size : window -> w:int -> h:int -> unit
(** {{:http://wiki.libsdl.org/SDL_SetWindowMinimumSize}
    SDL_SetWindowMinimumSize} *)

val set_window_position : window -> x:int -> y:int -> unit
(** {{:http://wiki.libsdl.org/SDL_SetWindowPosition}SDL_SetWindowPosition} *)

val set_window_size : window -> w:int -> h:int -> unit
(** {{:http://wiki.libsdl.org/SDL_SetWindowSize}SDL_SetWindowSize} *)

val set_window_title : window -> string -> unit
(** {{:http://wiki.libsdl.org/SDL_SetWindowTitle}SDL_SetWindowTitle} *)

val show_window : window -> unit
(** {{:http://wiki.libsdl.org/SDL_ShowWindow}SDL_ShowWindow} *)

val update_window_surface : window -> unit result
(** {{:http://wiki.libsdl.org/SDL_UpdateWindowSurface}
    SDL_UpdateWindowSurface} *)

val update_window_surface_rects : window -> rect list -> unit result
(** {{:http://wiki.libsdl.org/SDL_UpdateWindowSurfaceRects}
    SDL_UpdateWindowSurfaceRects} *)

val update_window_surface_rects_ba : window ->
  (int32, Bigarray.int32_elt) bigarray -> unit result
(** See {!update_window_surface_rects}. Each consecutive quadruplet defines a
    rectangle.

    @raise Invalid_argument if the length of the array is not
    a multiple of 4. *)

(** {2:opengl {{:http://wiki.libsdl.org/CategoryVideo}OpenGL contexts}} *)

type gl_context

(**/**)
val unsafe_gl_context_of_ptr : nativeint -> gl_context
val unsafe_ptr_of_gl_context : gl_context -> nativeint
(**/**)

(** SDL_GLContext *)

module Gl : sig
  (** {1:flags Context flags} *)

  type context_flags = int
  (** {{:http://wiki.libsdl.org/SDL_GLcontextFlag}SDL_GLcontextFlag} *)

  val context_debug_flag : context_flags
  val context_forward_compatible_flag : context_flags
  val context_robust_access_flag : context_flags
  val context_reset_isolation_flag : context_flags

  (** {1:profile Profile flags} *)

  type profile = int
  (** {{:http://wiki.libsdl.org/SDL_GLprofile}SDL_GLprofile} *)

  val context_profile_core : profile
  val context_profile_compatibility : profile
  val context_profile_es : profile

  (** {1:attr Attributes} *)

  type attr
  (** {{:http://wiki.libsdl.org/SDL_GLattr}SDL_GLattr} *)

  val red_size : attr
  val green_size : attr
  val blue_size : attr
  val alpha_size : attr
  val buffer_size : attr
  val doublebuffer : attr
  val depth_size : attr
  val stencil_size : attr
  val accum_red_size : attr
  val accum_green_size : attr
  val accum_blue_size : attr
  val accum_alpha_size : attr
  val stereo : attr
  val multisamplebuffers : attr
  val multisamplesamples : attr
  val accelerated_visual : attr
  val context_major_version : attr
  val context_minor_version : attr
  val context_egl : attr
  val context_flags : attr
  val context_profile_mask : attr
  val share_with_current_context : attr
  val framebuffer_srgb_capable : attr
end

val gl_create_context : window -> gl_context result
(** {{:http://wiki.libsdl.org/SDL_GL_CreateContext}SDL_GL_CreateContext} *)

val gl_bind_texture : texture -> (float * float) result
(** {{:http://wiki.libsdl.org/SDL_GL_BindTexture}SDL_GL_BindTexture} *)

val gl_delete_context : gl_context -> unit
(** {{:http://wiki.libsdl.org/SDL_GL_DeleteContext}SDL_GL_DeleteContext} *)

val gl_extension_supported : string -> bool
(** {{:http://wiki.libsdl.org/SDL_GL_ExtensionSupported}
    SDL_GL_ExtensionSupported} *)

val gl_get_attribute : Gl.attr -> int result
(** {{:http://wiki.libsdl.org/SDL_GL_GetAttribute}SDL_GL_GetAttribute} *)

val gl_get_current_context : unit -> gl_context result
(** {{:http://wiki.libsdl.org/SDL_GL_GetCurrentContext}
    SDL_GL_GetCurrentContext} *)

val gl_get_drawable_size : window -> int * int
(** {{:http://wiki.libsdl.org/SDL_GL_GetDrawableSize}SDL_GL_GetDrawableSize} *)

val gl_get_swap_interval : unit -> int result
(** {{:http://wiki.libsdl.org/SDL_GL_GetSwapInterval}SDL_GL_GetSwapInterval} *)

val gl_make_current : window -> gl_context -> unit result
(** {{:http://wiki.libsdl.org/SDL_GL_MakeCurrent}SDL_GL_MakeCurrent} *)

val gl_set_attribute : Gl.attr -> int -> unit result
(** {{:http://wiki.libsdl.org/SDL_GL_SetAttribute}SDL_GL_SetAttribute} *)

val gl_set_swap_interval : int -> unit result
(** {{:http://wiki.libsdl.org/SDL_GL_SetSwapInterval}SDL_GL_SetSwapInterval} *)

val gl_swap_window : window -> unit
(** {{:http://wiki.libsdl.org/SDL_GL_SwapWindow}SDL_GL_SwapWindow} *)

val gl_reset_attributes : unit -> unit
(** {{:http://wiki.libsdl.org/SDL_GL_ResetAttributes}SDL_GL_ResetAttributes}
    (SDL 2.0.2). *)

val gl_unbind_texture : texture -> unit result
(** {{:http://wiki.libsdl.org/SDL_GL_UnbindTexture}SDL_GL_UnbindTexture}
    {b Warning} Segfaults on SDL 2.0.1
    see {{:https://bugzilla.libsdl.org/show_bug.cgi?id=2296}this report}.*)

(** {2:screensaver Screen saver} *)

val disable_screen_saver : unit -> unit
(** {{:http://wiki.libsdl.org/SDL_DisableScreenSaver}SDL_DisableScreenSaver} *)

val enable_screen_saver : unit -> unit
(** {{:http://wiki.libsdl.org/SDL_EnableScreenSaver}SDL_EnableScreenSaver} *)

val is_screen_saver_enabled : unit -> bool
(** {{:http://wiki.libsdl.org/SDL_IsScreenSaverEnabled}
    SDL_IsScreenSaverEnabled} *)

(** {2:messageboxes Message boxes} *)

module Message_box : sig

  (** {1 Message box Buttons} *)

  type button_flags
  val button_returnkey_default : button_flags
  val button_escapekey_default : button_flags

  type button_data =
    { button_flags : button_flags;
      button_id : int;
      button_text : string }

  (** {1 Message box flags} *)

  type flags
  val error : flags
  val warning : flags
  val information : flags

  (** {1 Message box color scheme} *)

  type color = int * int * int
  (** r, g, b from 0 to 255 *)

  type color_scheme =
    { color_background : color;
      color_text : color;
      color_button_border : color;
      color_button_background : color;
      color_button_selected : color; }

  (** {1 Message box data} *)

  type data =
    { flags : flags;
      window : window option;
      title : string;
      message : string;
      buttons : button_data list;
      color_scheme : color_scheme option }
end

val show_message_box : Message_box.data -> int result
(** {{:https://wiki.libsdl.org/SDL_ShowMessageBox}SDL_ShowMessageBox} *)

val show_simple_message_box : Message_box.flags -> title:string -> string ->
  window option -> unit result
(** {{:https://wiki.libsdl.org/SDL_ShowSimpleMessageBox}
    SDL_ShowSimpleMessageBox} *)

(** {2:clipboard
    {{:http://wiki.libsdl.org/CategoryClipboard}Clipboard}} *)

val get_clipboard_text : unit -> string result
(** {{:http://wiki.libsdl.org/SDL_GetClipboardText}SDL_GetClipboardText} *)

val has_clipboard_text : unit -> bool
(** {{:http://wiki.libsdl.org/SDL_HasClipboardText}SDL_HasClipboardText} *)

val set_clipboard_text : string -> unit result
(** {{:http://wiki.libsdl.org/SDL_SetClipboardText}SDL_SetClipboardText} *)

(** {1:input Input} *)

type button_state
val pressed : button_state
val released : button_state

type toggle_state
val disable : toggle_state
val enable : toggle_state

(** {2:keyboard {{:http://wiki.libsdl.org/CategoryKeyboard}Keyboard}} *)

type scancode = int
(** {{:http://wiki.libsdl.org/SDL_Scancode}SDL_Scancode} *)

(** Constants and enumeration for {!scancode} *)
module Scancode : sig

  val enum : scancode ->
    [ `A | `Ac_back | `Ac_bookmarks | `Ac_forward | `Ac_home
    | `Ac_refresh | `Ac_search | `Ac_stop | `Again | `Alterase
    | `Apostrophe | `App1 | `App2 | `Application | `Audiomute
    | `Audionext | `Audioplay | `Audioprev | `Audiostop | `B
    | `Backslash | `Backspace | `Brightnessdown | `Brightnessup | `C
    | `Calculator | `Cancel | `Capslock | `Clear | `Clearagain | `Comma
    | `Computer | `Copy | `Crsel | `Currencysubunit | `Currencyunit
    | `Cut | `D | `Decimalseparator | `Delete | `Displayswitch | `Down
    | `E | `Eject | `End | `Equals | `Escape | `Execute | `Exsel | `F
    | `F1 | `F10 | `F11 | `F12 | `F13 | `F14 | `F15 | `F16 | `F17 | `F18
    | `F19 | `F2 | `F20 | `F21 | `F22 | `F23 | `F24 | `F3 | `F4 | `F5
    | `F6 | `F7 | `F8 | `F9 | `Find | `G | `Grave | `H | `Help | `Home
    | `I | `Insert | `International1 | `International2
    | `International3 | `International4 | `International5
    | `International6 | `International7 | `International8
    | `International9 | `J | `K | `K0 | `K1 | `K2 | `K3 | `K4 | `K5
    | `K6 | `K7 | `K8 | `K9 | `Kbdillumdown | `Kbdillumtoggle
    | `Kbdillumup | `Kp_0 | `Kp_00 | `Kp_000 | `Kp_1 | `Kp_2 | `Kp_3
    | `Kp_4 | `Kp_5 | `Kp_6 | `Kp_7 | `Kp_8 | `Kp_9 | `Kp_a
    | `Kp_ampersand | `Kp_at | `Kp_b | `Kp_backspace | `Kp_binary
    | `Kp_c | `Kp_clear | `Kp_clearentry | `Kp_colon | `Kp_comma | `Kp_d
    | `Kp_dblampersand | `Kp_dblverticalbar | `Kp_decimal | `Kp_divide
    | `Kp_e | `Kp_enter | `Kp_equals | `Kp_equalsas400 | `Kp_exclam
    | `Kp_f | `Kp_greater | `Kp_hash | `Kp_hexadecimal | `Kp_leftbrace
    | `Kp_leftparen | `Kp_less | `Kp_memadd | `Kp_memclear
    | `Kp_memdivide | `Kp_memmultiply | `Kp_memrecall | `Kp_memstore
    | `Kp_memsubtract | `Kp_minus | `Kp_multiply | `Kp_octal
    | `Kp_percent | `Kp_period | `Kp_plus | `Kp_plusminus | `Kp_power
    | `Kp_rightbrace | `Kp_rightparen | `Kp_space | `Kp_tab
    | `Kp_verticalbar | `Kp_xor | `L | `Lalt | `Lang1 | `Lang2 | `Lang3
    | `Lang4 | `Lang5 | `Lang6 | `Lang7 | `Lang8 | `Lang9 | `Lctrl
    | `Left | `Leftbracket | `Lgui | `Lshift | `M | `Mail | `Mediaselect
    | `Menu | `Minus | `Mode | `Mute | `N | `Nonusbackslash
    | `Nonushash | `Numlockclear | `O | `Oper | `Out | `P | `Pagedown
    | `Pageup | `Paste | `Pause | `Period | `Power | `Printscreen
    | `Prior | `Q | `R | `Ralt | `Rctrl | `Return | `Return2 | `Rgui
    | `Right | `Rightbracket | `Rshift | `S | `Scrolllock | `Select
    | `Semicolon | `Separator | `Slash | `Sleep | `Space | `Stop
    | `Sysreq | `T | `Tab | `Thousandsseparator | `U | `Undo | `Unknown
    | `Up | `V | `Volumedown | `Volumeup | `W | `Www | `X | `Y | `Z ]

  val num_scancodes : int
  val unknown : scancode
  val a : scancode
  val b : scancode
  val c : scancode
  val d : scancode
  val e : scancode
  val f : scancode
  val g : scancode
  val h : scancode
  val i : scancode
  val j : scancode
  val k : scancode
  val l : scancode
  val m : scancode
  val n : scancode
  val o : scancode
  val p : scancode
  val q : scancode
  val r : scancode
  val s : scancode
  val t : scancode
  val u : scancode
  val v : scancode
  val w : scancode
  val x : scancode
  val y : scancode
  val z : scancode
  val k1 : scancode
  val k2 : scancode
  val k3 : scancode
  val k4 : scancode
  val k5 : scancode
  val k6 : scancode
  val k7 : scancode
  val k8 : scancode
  val k9 : scancode
  val k0 : scancode
  val return : scancode
  val escape : scancode
  val backspace : scancode
  val tab : scancode
  val space : scancode
  val minus : scancode
  val equals : scancode
  val leftbracket : scancode
  val rightbracket : scancode
  val backslash : scancode
  val nonushash : scancode
  val semicolon : scancode
  val apostrophe : scancode
  val grave : scancode
  val comma : scancode
  val period : scancode
  val slash : scancode
  val capslock : scancode
  val f1 : scancode
  val f2 : scancode
  val f3 : scancode
  val f4 : scancode
  val f5 : scancode
  val f6 : scancode
  val f7 : scancode
  val f8 : scancode
  val f9 : scancode
  val f10 : scancode
  val f11 : scancode
  val f12 : scancode
  val printscreen : scancode
  val scrolllock : scancode
  val pause : scancode
  val insert : scancode
  val home : scancode
  val pageup : scancode
  val delete : scancode
  val kend : scancode
  val pagedown : scancode
  val right : scancode
  val left : scancode
  val down : scancode
  val up : scancode
  val numlockclear : scancode
  val kp_divide : scancode
  val kp_multiply : scancode
  val kp_minus : scancode
  val kp_plus : scancode
  val kp_enter : scancode
  val kp_1 : scancode
  val kp_2 : scancode
  val kp_3 : scancode
  val kp_4 : scancode
  val kp_5 : scancode
  val kp_6 : scancode
  val kp_7 : scancode
  val kp_8 : scancode
  val kp_9 : scancode
  val kp_0 : scancode
  val kp_period : scancode
  val nonusbackslash : scancode
  val application : scancode
  val kp_equals : scancode
  val f13 : scancode
  val f14 : scancode
  val f15 : scancode
  val f16 : scancode
  val f17 : scancode
  val f18 : scancode
  val f19 : scancode
  val f20 : scancode
  val f21 : scancode
  val f22 : scancode
  val f23 : scancode
  val f24 : scancode
  val execute : scancode
  val help : scancode
  val menu : scancode
  val select : scancode
  val stop : scancode
  val again : scancode
  val undo : scancode
  val cut : scancode
  val copy : scancode
  val paste : scancode
  val find : scancode
  val mute : scancode
  val volumeup : scancode
  val volumedown : scancode
  val kp_comma : scancode
  val kp_equalsas400 : scancode
  val international1 : scancode
  val international2 : scancode
  val international3 : scancode
  val international4 : scancode
  val international5 : scancode
  val international6 : scancode
  val international7 : scancode
  val international8 : scancode
  val international9 : scancode
  val lang1 : scancode
  val lang2 : scancode
  val lang3 : scancode
  val lang4 : scancode
  val lang5 : scancode
  val lang6 : scancode
  val lang7 : scancode
  val lang8 : scancode
  val lang9 : scancode
  val alterase : scancode
  val sysreq : scancode
  val cancel : scancode
  val clear : scancode
  val prior : scancode
  val return2 : scancode
  val separator : scancode
  val out : scancode
  val oper : scancode
  val clearagain : scancode
  val crsel : scancode
  val exsel : scancode
  val kp_00 : scancode
  val kp_000 : scancode
  val thousandsseparator : scancode
  val decimalseparator : scancode
  val currencyunit : scancode
  val currencysubunit : scancode
  val kp_leftparen : scancode
  val kp_rightparen : scancode
  val kp_leftbrace : scancode
  val kp_rightbrace : scancode
  val kp_tab : scancode
  val kp_backspace : scancode
  val kp_a : scancode
  val kp_b : scancode
  val kp_c : scancode
  val kp_d : scancode
  val kp_e : scancode
  val kp_f : scancode
  val kp_xor : scancode
  val kp_power : scancode
  val kp_percent : scancode
  val kp_less : scancode
  val kp_greater : scancode
  val kp_ampersand : scancode
  val kp_dblampersand : scancode
  val kp_verticalbar : scancode
  val kp_dblverticalbar : scancode
  val kp_colon : scancode
  val kp_hash : scancode
  val kp_space : scancode
  val kp_at : scancode
  val kp_exclam : scancode
  val kp_memstore : scancode
  val kp_memrecall : scancode
  val kp_memclear : scancode
  val kp_memadd : scancode
  val kp_memsubtract : scancode
  val kp_memmultiply : scancode
  val kp_memdivide : scancode
  val kp_plusminus : scancode
  val kp_clear : scancode
  val kp_clearentry : scancode
  val kp_binary : scancode
  val kp_octal : scancode
  val kp_decimal : scancode
  val kp_hexadecimal : scancode
  val lctrl : scancode
  val lshift : scancode
  val lalt : scancode
  val lgui : scancode
  val rctrl : scancode
  val rshift : scancode
  val ralt : scancode
  val rgui : scancode
  val mode : scancode
  val audionext : scancode
  val audioprev : scancode
  val audiostop : scancode
  val audioplay : scancode
  val audiomute : scancode
  val mediaselect : scancode
  val www : scancode
  val mail : scancode
  val calculator : scancode
  val computer : scancode
  val ac_search : scancode
  val ac_home : scancode
  val ac_back : scancode
  val ac_forward : scancode
  val ac_stop : scancode
  val ac_refresh : scancode
  val ac_bookmarks : scancode
  val brightnessdown : scancode
  val brightnessup : scancode
  val displayswitch : scancode
  val kbdillumtoggle : scancode
  val kbdillumdown : scancode
  val kbdillumup : scancode
  val eject : scancode
  val sleep : scancode
  val app1 : scancode
  val app2 : scancode
end

type keycode = int
(** {{:http://wiki.libsdl.org/SDL_Keycode}SDL_Keycode} *)

(** Constants for {!keycode} *)
module K : sig
  val scancode_mask : int
  val unknown : keycode
  val return : keycode
  val escape : keycode
  val backspace : keycode
  val tab : keycode
  val space : keycode
  val exclaim : keycode
  val quotedbl : keycode
  val hash : keycode
  val percent : keycode
  val dollar : keycode
  val ampersand : keycode
  val quote : keycode
  val leftparen : keycode
  val rightparen : keycode
  val asterisk : keycode
  val plus : keycode
  val comma : keycode
  val minus : keycode
  val period : keycode
  val slash : keycode
  val k0 : keycode
  val k1 : keycode
  val k2 : keycode
  val k3 : keycode
  val k4 : keycode
  val k5 : keycode
  val k6 : keycode
  val k7 : keycode
  val k8 : keycode
  val k9 : keycode
  val colon : keycode
  val semicolon : keycode
  val less : keycode
  val equals : keycode
  val greater : keycode
  val question : keycode
  val at : keycode
  val leftbracket : keycode
  val backslash : keycode
  val rightbracket : keycode
  val caret : keycode
  val underscore : keycode
  val backquote : keycode
  val a : keycode
  val b : keycode
  val c : keycode
  val d : keycode
  val e : keycode
  val f : keycode
  val g : keycode
  val h : keycode
  val i : keycode
  val j : keycode
  val k : keycode
  val l : keycode
  val m : keycode
  val n : keycode
  val o : keycode
  val p : keycode
  val q : keycode
  val r : keycode
  val s : keycode
  val t : keycode
  val u : keycode
  val v : keycode
  val w : keycode
  val x : keycode
  val y : keycode
  val z : keycode
  val capslock : keycode
  val f1 : keycode
  val f2 : keycode
  val f3 : keycode
  val f4 : keycode
  val f5 : keycode
  val f6 : keycode
  val f7 : keycode
  val f8 : keycode
  val f9 : keycode
  val f10 : keycode
  val f11 : keycode
  val f12 : keycode
  val printscreen : keycode
  val scrolllock : keycode
  val pause : keycode
  val insert : keycode
  val home : keycode
  val pageup : keycode
  val delete : keycode
  val kend : keycode
  val pagedown : keycode
  val right : keycode
  val left : keycode
  val down : keycode
  val up : keycode
  val numlockclear : keycode
  val kp_divide : keycode
  val kp_multiply : keycode
  val kp_minus : keycode
  val kp_plus : keycode
  val kp_enter : keycode
  val kp_1 : keycode
  val kp_2 : keycode
  val kp_3 : keycode
  val kp_4 : keycode
  val kp_5 : keycode
  val kp_6 : keycode
  val kp_7 : keycode
  val kp_8 : keycode
  val kp_9 : keycode
  val kp_0 : keycode
  val kp_period : keycode
  val application : keycode
  val power : keycode
  val kp_equals : keycode
  val f13 : keycode
  val f14 : keycode
  val f15 : keycode
  val f16 : keycode
  val f17 : keycode
  val f18 : keycode
  val f19 : keycode
  val f20 : keycode
  val f21 : keycode
  val f22 : keycode
  val f23 : keycode
  val f24 : keycode
  val execute : keycode
  val help : keycode
  val menu : keycode
  val select : keycode
  val stop : keycode
  val again : keycode
  val undo : keycode
  val cut : keycode
  val copy : keycode
  val paste : keycode
  val find : keycode
  val mute : keycode
  val volumeup : keycode
  val volumedown : keycode
  val kp_comma : keycode
  val kp_equalsas400 : keycode
  val alterase : keycode
  val sysreq : keycode
  val cancel : keycode
  val clear : keycode
  val prior : keycode
  val return2 : keycode
  val separator : keycode
  val out : keycode
  val oper : keycode
  val clearagain : keycode
  val crsel : keycode
  val exsel : keycode
  val kp_00 : keycode
  val kp_000 : keycode
  val thousandsseparator : keycode
  val decimalseparator : keycode
  val currencyunit : keycode
  val currencysubunit : keycode
  val kp_leftparen : keycode
  val kp_rightparen : keycode
  val kp_leftbrace : keycode
  val kp_rightbrace : keycode
  val kp_tab : keycode
  val kp_backspace : keycode
  val kp_a : keycode
  val kp_b : keycode
  val kp_c : keycode
  val kp_d : keycode
  val kp_e : keycode
  val kp_f : keycode
  val kp_xor : keycode
  val kp_power : keycode
  val kp_percent : keycode
  val kp_less : keycode
  val kp_greater : keycode
  val kp_ampersand : keycode
  val kp_dblampersand : keycode
  val kp_verticalbar : keycode
  val kp_dblverticalbar : keycode
  val kp_colon : keycode
  val kp_hash : keycode
  val kp_space : keycode
  val kp_at : keycode
  val kp_exclam : keycode
  val kp_memstore : keycode
  val kp_memrecall : keycode
  val kp_memclear : keycode
  val kp_memadd : keycode
  val kp_memsubtract : keycode
  val kp_memmultiply : keycode
  val kp_memdivide : keycode
  val kp_plusminus : keycode
  val kp_clear : keycode
  val kp_clearentry : keycode
  val kp_binary : keycode
  val kp_octal : keycode
  val kp_decimal : keycode
  val kp_hexadecimal : keycode
  val lctrl : keycode
  val lshift : keycode
  val lalt : keycode
  val lgui : keycode
  val rctrl : keycode
  val rshift : keycode
  val ralt : keycode
  val rgui : keycode
  val mode : keycode
  val audionext : keycode
  val audioprev : keycode
  val audiostop : keycode
  val audioplay : keycode
  val audiomute : keycode
  val mediaselect : keycode
  val www : keycode
  val mail : keycode
  val calculator : keycode
  val computer : keycode
  val ac_search : keycode
  val ac_home : keycode
  val ac_back : keycode
  val ac_forward : keycode
  val ac_stop : keycode
  val ac_refresh : keycode
  val ac_bookmarks : keycode
  val brightnessdown : keycode
  val brightnessup : keycode
  val displayswitch : keycode
  val kbdillumtoggle : keycode
  val kbdillumdown : keycode
  val kbdillumup : keycode
  val eject : keycode
  val sleep : keycode
end

type keymod = int
(** {{:http://wiki.libsdl.org/SDL_Keymod}SDL_Keymod}. *)

(** Constants for {!keymod} *)
module Kmod : sig
  val none : keymod
  val lshift : keymod
  val rshift : keymod
  val lctrl : keymod
  val rctrl : keymod
  val lalt : keymod
  val ralt : keymod
  val lgui : keymod
  val rgui : keymod
  val num : keymod
  val caps : keymod
  val mode : keymod
  val reserved : keymod
  val ctrl : keymod
  val shift : keymod
  val alt : keymod
  val gui : keymod
end

val get_keyboard_focus : unit -> window option
(** {{:http://wiki.libsdl.org/SDL_GetKeyboardFocus}
    SDL_GetKeyboardFocus} *)

val get_keyboard_state : unit -> (int, Bigarray.int8_unsigned_elt) bigarray
(** {{:http://wiki.libsdl.org/SDL_GetKeyboardState}SDL_GetKeyboardState} *)

val get_key_from_name : string -> keycode
(** {{:http://wiki.libsdl.org/SDL_GetKeyFromName}SDL_GetKeyFromName} *)

val get_key_from_scancode : scancode -> keycode
(** {{:http://wiki.libsdl.org/SDL_GetKeyFromScancode}SDL_GetKeyFromScancode} *)

val get_key_name : keycode -> string
(** {{:http://wiki.libsdl.org/SDL_GetKeyName}SDL_GetKeyName} *)

val get_mod_state : unit -> keymod
(** {{:http://wiki.libsdl.org/SDL_GetModState}SDL_GetModState} *)

val get_scancode_from_key : keycode -> scancode
(** {{:http://wiki.libsdl.org/SDL_GetScancodeFromKey}SDL_GetScancodeFromKey} *)

val get_scancode_from_name : string -> scancode
(** {{:http://wiki.libsdl.org/SDL_GetScancodeFromName}SDL_GetScancodeFromName}*)

val get_scancode_name : scancode -> string
(** {{:http://wiki.libsdl.org/SDL_GetScancodeName}SDL_GetScancodeName} *)

val has_screen_keyboard_support : unit -> bool
(** {{:http://wiki.libsdl.org/SDL_HasScreenKeyboardSupport}
    SDL_HasScreenKeyboardSupport} *)

val is_screen_keyboard_shown : window -> bool
(** {{:http://wiki.libsdl.org/SDL_IsScreenKeyboardShown}
    SDL_IsScreenKeyboardShown} *)

val is_text_input_active : unit -> bool
(** {{:http://wiki.libsdl.org/SDL_IsTextInputActive}SDL_IsTextInputActive} *)

val set_mod_state : keymod -> unit
(** {{:http://wiki.libsdl.org/SDL_SetModState}SDL_SetModState} *)

val set_text_input_rect : rect option -> unit
(** {{:http://wiki.libsdl.org/SDL_SetTextInputRect}SDL_SetTextInputRect} *)

val start_text_input : unit -> unit
(** {{:http://wiki.libsdl.org/SDL_StartTextInput}SDL_StartTextInput} *)

val stop_text_input : unit -> unit
(** {{:http://wiki.libsdl.org/SDL_StopTextInput}SDL_StopTextInput} *)

(** {2:mouse {{:http://wiki.libsdl.org/CategoryMouse}Mouse}} *)

type cursor

(**/**)
val unsafe_cursor_of_ptr : nativeint -> cursor
val unsafe_ptr_of_cursor : cursor -> nativeint
(**/**)

(** SDL_Cursor *)

module System_cursor : sig
  type t
  val arrow : t
  val ibeam : t
  val wait : t
  val crosshair : t
  val waitarrow : t
  val size_nw_se : t
  val size_ne_sw : t
  val size_we : t
  val size_ns : t
  val size_all : t
  val no : t
  val hand : t
end

module Button : sig
  val left : int
  val middle : int
  val right : int
  val x1 : int
  val x2 : int

  val lmask : uint32
  val mmask : uint32
  val rmask : uint32
  val x1mask : uint32
  val x2mask : uint32
end

val create_color_cursor : surface -> hot_x:int -> hot_y:int -> cursor result
(** {{:http://wiki.libsdl.org/SDL_CreateColorCursor}SDL_CreateColorCursor} *)

val create_cursor : (int, Bigarray.int8_unsigned_elt) bigarray ->
  (int, Bigarray.int8_unsigned_elt) bigarray -> w:int -> h:int -> hot_x:int ->
  hot_y:int -> cursor result
(** {{:http://wiki.libsdl.org/SDL_CreateCursor}SDL_CreateCursor} *)

val create_system_cursor : System_cursor.t -> cursor result
(** {{:https://wiki.libsdl.org/SDL_CreateSystemCursor}SDL_CreateSystemCursor} *)

val free_cursor : cursor -> unit
(** {{:http://wiki.libsdl.org/SDL_FreeCursor}SDL_FreeCursor} *)

val get_cursor : unit -> cursor option
(** {{:http://wiki.libsdl.org/SDL_GetCursor}SDL_GetCursor} *)

val get_default_cursor : unit -> cursor option
(** {{:http://wiki.libsdl.org/SDL_GetDefaultCursor}SDL_GetDefaultCursor} *)

val get_mouse_focus : unit -> window option
(** {{:http://wiki.libsdl.org/SDL_GetMouseFocus}SDL_GetMouseFocus} *)

val get_mouse_state : unit -> uint32 * (int * int)
(** {{:http://wiki.libsdl.org/SDL_GetMouseState}SDL_GetMouseState} *)

val get_relative_mouse_mode : unit -> bool
(** {{:http://wiki.libsdl.org/SDL_GetRelativeMouseMode}
    SDL_GetRelativeMouseMode} *)

val get_relative_mouse_state : unit -> uint32 * (int * int)
(** {{:http://wiki.libsdl.org/SDL_GetRelativeMouseState}
    SDL_GetRelativeMouseState} *)

val get_cursor_shown : unit -> bool result
(** {{:http://wiki.libsdl.org/SDL_ShowCursor}SDL_ShowCursor} with
    SDL_QUERY. *)

val set_cursor : cursor option -> unit
(** {{:http://wiki.libsdl.org/SDL_SetCursor}SDL_SetCursor} *)

val set_relative_mouse_mode : bool -> unit result
(** {{:http://wiki.libsdl.org/SDL_SetRelativeMouseMode}
    SDL_SetRelativeMouseMode} *)

val show_cursor : bool -> bool result
(** {{:http://wiki.libsdl.org/SDL_ShowCursor}SDL_ShowCursor}. See also
    {!get_cursor_shown}. *)

val warp_mouse_in_window : window option -> x:int -> y:int -> unit
(** {{:http://wiki.libsdl.org/SDL_WarpMouseInWindow}SDL_WarpMouseInWindow} *)

(** {2:touch Touch and gestures} *)

type touch_id = int64
(** SDL_TouchID *)

val touch_mouse_id : touch_id
(** SDL_TOUCH_MOUSEID *)

type gesture_id = int64
(** SDL_GestureID *)

type finger_id = int64
(** SDL_FingerID *)

type finger
(** SDL_Finger *)

module Finger : sig
  val id : finger -> finger_id
  val x : finger -> float
  val y : finger -> float
  val pressure : finger -> float
end

val get_num_touch_devices : unit -> int
(** {{:https://wiki.libsdl.org/SDL_GetNumTouchDevices}SDL_GetNumTouchDevices}.*)

val get_num_touch_fingers : touch_id -> int
(** {{:https://wiki.libsdl.org/SDL_GetNumTouchFingers}SDL_GetNumTouchFingers}.*)

val get_touch_device : int -> touch_id result
(** {{:https://wiki.libsdl.org/SDL_GetTouchDevice}SDL_GetTouchDevice}.*)

val get_touch_finger : touch_id -> int -> finger option
(** {{:https://wiki.libsdl.org/SDL_GetTouchFinger}SDL_GetTouchFinger}.*)

val load_dollar_templates : touch_id -> rw_ops -> unit result
(** {{:https://wiki.libsdl.org/SDL_LoadDollarTemplates}
    SDL_LoadDollarTemplates} *)

val record_gesture : touch_id -> unit result
(** {{:https://wiki.libsdl.org/SDL_RecordGesture}SDL_RecordGesture}.*)

val save_dollar_template : gesture_id -> rw_ops -> unit result
(** {{:https://wiki.libsdl.org/SDL_SaveDollarTemplate}SDL_SaveDollarTemplate}.*)

val save_all_dollar_templates : rw_ops -> unit result
(** {{:https://wiki.libsdl.org/SDL_SaveAllDollarTemplate}
    SDL_SaveAllDollarTemplate}.*)

(** {2:joystick {{:http://wiki.libsdl.org/CategoryJoystick}Joystick}} *)

type joystick_guid
(** SDL_JoystickGUID. *)

type joystick_id = int32
(** SDL_JoystickID *)

type joystick

(**/**)
val unsafe_joystick_of_ptr : nativeint -> joystick
val unsafe_ptr_of_joystick : joystick -> nativeint
(**/**)

(** SDL_Joystick *)

module Hat : sig
  type t = int
  val centered : int
  val up : int
  val right : int
  val down : int
  val left : int
  val rightup : int
  val rightdown : int
  val leftup : int
  val leftdown : int
end

val joystick_close : joystick -> unit
(** {{:http://wiki.libsdl.org/SDL_JoystickClose}SDL_JoystickClose} *)

val joystick_get_event_state : unit -> toggle_state result
(** {{:http://wiki.libsdl.org/SDL_JoystickEventState}
    SDL_JoystickEventState} with SDL_QUERY. *)

val joystick_set_event_state : toggle_state -> toggle_state result
(** {{:http://wiki.libsdl.org/SDL_JoystickEventState}
    SDL_JoystickEventState}. See also {!joystick_get_event_state}. *)

val joystick_get_attached : joystick -> bool
(** {{:https://wiki.libsdl.org/SDL_JoystickGetAttached}
    SDL_JoystickGetAttached} *)

val joystick_get_axis : joystick -> int -> int16
(** {{:http://wiki.libsdl.org/SDL_JoystickGetAxis}SDL_JoystickGetAxis} *)

val joystick_get_ball : joystick -> int -> (int * int) result
(** {{:http://wiki.libsdl.org/SDL_JoystickGetBall}SDL_JoystickGetBall} *)

val joystick_get_button : joystick -> int -> uint8
(** {{:http://wiki.libsdl.org/SDL_JoystickGetButton}SDL_JoystickGetButton} *)

val joystick_get_device_guid : int -> joystick_guid
(** {{:http://wiki.libsdl.org/SDL_JoystickGetDeviceGUID}
    SDL_JoystickGetDeviceGUID} *)

val joystick_get_guid : joystick -> joystick_guid
(** {{:http://wiki.libsdl.org/SDL_JoystickGetGUID}SDL_JoystickGetGUID} *)

val joystick_get_guid_from_string : string -> joystick_guid
(** {{:http://wiki.libsdl.org/SDL_JoystickGetGUIDFromString}
    SDL_JoystickGetGUIDFromString} *)

val joystick_get_guid_string : joystick_guid -> string
(** {{:http://wiki.libsdl.org/SDL_JoystickGetGUIDString}
    SDL_JoystickGetGUIDString} *)

val joystick_get_hat : joystick -> int -> Hat.t
(** {{:http://wiki.libsdl.org/SDL_JoystickGetHat}SDL_JoystickGetHat} *)

val joystick_instance_id : joystick -> joystick_id result
(** {{:http://wiki.libsdl.org/SDL_JoystickInstanceID}SDL_JoystickInstanceID} *)

val joystick_name : joystick -> string result
(** {{:http://wiki.libsdl.org/SDL_JoystickName}SDL_JoystickName} *)

val joystick_name_for_index : int -> string result
(** {{:http://wiki.libsdl.org/SDL_JoystickNameForIndex}
    SDL_JoystickNameForIndex} *)

val joystick_num_axes : joystick -> int result
(** {{:http://wiki.libsdl.org/SDL_JoystickNumAxes}SDL_JoystickNumAxes} *)

val joystick_num_balls : joystick -> int result
(** {{:http://wiki.libsdl.org/SDL_JoystickNumBalls}SDL_JoystickNumBalls} *)

val joystick_num_buttons : joystick -> int result
(** {{:http://wiki.libsdl.org/SDL_JoystickNumButtons}SDL_JoystickNumButtons} *)

val joystick_num_hats : joystick -> int result
(** {{:http://wiki.libsdl.org/SDL_JoystickNumHats}SDL_JoystickNumHats} *)

val joystick_open : int -> joystick result
(** {{:http://wiki.libsdl.org/SDL_JoystickOpen}SDL_JoystickOpen} *)

val joystick_update : unit -> unit
(** {{:http://wiki.libsdl.org/SDL_JoystickUpdate}SDL_JoystickUpdate} *)

val num_joysticks : unit -> int result
(** {{:http://wiki.libsdl.org/SDL_NumJoysticks}SDL_NumJoysticks} *)

(** {2:gamecontroller
  {{:http://wiki.libsdl.org/CategoryGameController}Game controller}} *)

type game_controller

(**/**)
val unsafe_game_controller_of_ptr : nativeint -> game_controller
val unsafe_ptr_of_game_controller : game_controller -> nativeint
(**/**)

(** SDL_GameController *)

module Controller : sig
  type bind_type
  val bind_type_none : bind_type
  val bind_type_button : bind_type
  val bind_type_axis : bind_type
  val bind_type_hat : bind_type

  type axis
  val axis_invalid : axis
  val axis_left_x : axis
  val axis_left_y : axis
  val axis_right_x : axis
  val axis_right_y : axis
  val axis_trigger_left : axis
  val axis_trigger_right : axis
  val axis_max : axis

  type button
  val button_invalid : button
  val button_a : button
  val button_b : button
  val button_x : button
  val button_y : button
  val button_back : button
  val button_guide : button
  val button_start : button
  val button_left_stick : button
  val button_right_stick : button
  val button_left_shoulder : button
  val button_right_shoulder : button
  val button_dpad_up : button
  val button_dpad_down : button
  val button_dpad_left : button
  val button_dpad_right : button
  val button_max : button

  type button_bind
  (** SDL_GameControllerButtonBind *)
  val bind_type : button_bind -> bind_type
  val bind_button_value : button_bind -> int
  val bind_axis_value : button_bind -> int
  val bind_hat_value : button_bind -> int * int
end

val game_controller_add_mapping : string -> bool result
(**  {{:http://wiki.libsdl.org/SDL_GameControllerAddMapping}
     SDL_GameControllerAddMapping} *)

val game_controller_add_mapping_from_file : string -> int result
(** {{:https://wiki.libsdl.org/SDL_GameControllerAddMappingsFromFile}
    SDL_GameControllerAddMappingsFromFile} (SDL 2.0.2). *)

val game_controller_add_mapping_from_rw : rw_ops -> bool -> int result
(** {{:https://wiki.libsdl.org/SDL_GameControllerAddMappingsFromRW}
    SDL_GameControllerAddMappingsFromFile} (SDL 2.0.2). *)

val game_controller_close : game_controller -> unit
(**  {{:http://wiki.libsdl.org/SDL_GameControllerClose}
     SDL_GameControllerClose} *)

val game_controller_get_event_state : unit -> toggle_state result
(**  {{:http://wiki.libsdl.org/SDL_GameControllerEventState}
     SDL_GameControllerEventState} with SDL_QUERY *)

val game_controller_set_event_state : toggle_state -> toggle_state result
(**  {{:http://wiki.libsdl.org/SDL_GameControllerEventState}
     SDL_GameControllerEventState}.
     See also {!game_controller_get_event_state}. *)

val game_controller_get_attached : game_controller -> bool
(**  {{:http://wiki.libsdl.org/SDL_GameControllerGetAttached}
     SDL_GameControllerGetAttached} *)

val game_controller_get_axis : game_controller -> Controller.axis -> int16
(**  {{:http://wiki.libsdl.org/SDL_GameControllerGetAxis}
     SDL_GameControllerGetAxis} *)

val game_controller_get_axis_from_string : string -> Controller.axis
(**  {{:http://wiki.libsdl.org/SDL_GameControllerGetAxisFromString}
     SDL_GameControllerGetAxisFromString} *)

val game_controller_get_bind_for_axis : game_controller -> Controller.axis ->
  Controller.button_bind
(**  {{:http://wiki.libsdl.org/SDL_GameControllerGetBindForAxis}
     SDL_GameControllerGetBindForAxis} *)

val game_controller_get_bind_for_button : game_controller ->
  Controller.button -> Controller.button_bind
(**  {{:http://wiki.libsdl.org/SDL_GameControllerGetBindForButton}
     SDL_GameControllerGetBindForButton} *)

val game_controller_get_button : game_controller -> Controller.button -> uint8
(**  {{:http://wiki.libsdl.org/SDL_GameControllerGetButton}
     SDL_GameControllerGetButton} *)

val game_controller_get_button_from_string : string -> Controller.button
(**  {{:http://wiki.libsdl.org/SDL_GameControllerGetButtonFromString}
     SDL_GameControllerGetButtonFromString} *)

val game_controller_get_joystick : game_controller -> joystick result
(**  {{:http://wiki.libsdl.org/SDL_GameControllerGetJoystick}
     SDL_GameControllerGetJoystick} *)

val game_controller_get_string_for_axis : Controller.axis -> string option
(**  {{:http://wiki.libsdl.org/SDL_GameControllerGetStringForAxis}
     SDL_GameControllerGetStringForAxis} *)

val game_controller_get_string_for_button : Controller.button -> string option
(**  {{:http://wiki.libsdl.org/SDL_GameControllerGetStringForButton}
     SDL_GameControllerGetStringForButton} *)

val game_controller_mapping : game_controller -> string result
(**  {{:http://wiki.libsdl.org/SDL_GameControllerMapping}
     SDL_GameControllerMapping} *)

val game_controller_mapping_for_guid : joystick_guid -> string result
(**  {{:http://wiki.libsdl.org/SDL_GameControllerMappingForGUID}
     SDL_GameControllerMappingForGUID} *)

val game_controller_name : game_controller -> string result
(**  {{:http://wiki.libsdl.org/SDL_GameControllerName}SDL_GameControllerName} *)

val game_controller_name_for_index : int -> string result
(**  {{:http://wiki.libsdl.org/SDL_GameControllerNameForIndex}
     SDL_GameControllerNameForIndex} *)

val game_controller_open : int -> game_controller result
(**  {{:http://wiki.libsdl.org/SDL_GameControllerOpen}
     SDL_GameControllerOpen} *)

val game_controller_update : unit -> unit
(**  {{:http://wiki.libsdl.org/SDL_GameControllerUpdate}
     SDL_GameControllerUpdate} *)

val is_game_controller : int -> bool
(**  {{:http://wiki.libsdl.org/SDL_IsGameController}SDL_IsGameController} *)

(** {2:events {{:http://wiki.libsdl.org/CategoryEvents}Events}} *)

type event_type = int
(** {{:http://wiki.libsdl.org/SDL_EventType}SDL_EventType}.
    See {!Event} for constants. *)

type event
(** {{:http://wiki.libsdl.org/SDL_Event}SDL_Event} *)

(** {!event} accessors and {!event_type} constants and {{!enum}enumeration}. *)
module Event : sig

  (** {1:event Event}

    Once you have determined the {!typ} you can access fields
    available for that type. Safe if you use the wrong accessors:
    you will just end with garbage data.  *)

  type 'b field
  (** The type for event fields. *)

  val create : unit -> event
  (** [create ()] is an uninitialized event structure. *)

  val get : event -> 'b field -> 'b
  (** [get e f] gets the field [f] of [e]. *)

  val set : event -> 'b field -> 'b -> unit
  (** [set e f v] sets the field [f] of [e] to [v]. *)

  (** {1 Event types and their fields}

      {ul
      {- {!common}}
      {- {!application}}
      {- {!clipboard}}
      {- {!controller}}
      {- {!dollar}}
      {- {!drop}}
      {- {!touch}}
      {- {!joystickev}}
      {- {!keyboard}}
      {- {!mouse}}
      {- {!multigestureev}}
      {- {!quitev}}
      {- {!syswm}}
      {- {!text}}
      {- {!window}}} *)

  (** {2 Event type aliases and misc} *)

  val first_event : event_type
  val last_event : event_type

  (** {2:common Common}

      These fields are common to all event types.  *)

  val typ : event_type field
  val timestamp : uint32 field

  (** {2:application Application events} *)

  val app_did_enter_background : event_type
  val app_did_enter_foreground : event_type
  val app_low_memory : event_type
  val app_terminating : event_type
  val app_will_enter_background : event_type
  val app_will_enter_foreground : event_type

  (** {2:clipboard Clipboard} *)

  val clipboard_update : event_type

  (** {2:controller Controller events} *)

  val controller_axis_motion : event_type
  val controller_button_down : event_type
  val controller_button_up : event_type
  val controller_device_added : event_type
  val controller_device_remapped : event_type
  val controller_device_removed : event_type

  (** {3 {{:http://wiki.libsdl.org/SDL_ControllerAxisEvent}
      SDL_ControllerAxisEvent} fields} *)

  val controller_axis_which : joystick_id field
  val controller_axis_axis : uint8 field
  val controller_axis_value : int16 field

  (** {3 {{:http://wiki.libsdl.org/SDL_ControllerButtonEvent}
      SDL_ControllerButtonEvent} fields} *)

  val controller_button_which : joystick_id field
  val controller_button_button : uint8 field
  val controller_button_state : button_state field

  (** {3 {{:http://wiki.libsdl.org/SDL_ControllerDeviceEvent}
      SDL_ControllerDeviceEvent} fields} *)

  val controller_device_which : joystick_id field

  (** {2:dollar Dollar gesture events} *)

  val dollar_gesture : event_type
  val dollar_record : event_type

  (** {3 {{:http://wiki.libsdl.org/SDL_DollarGestureEvent}
      SDL_DollarGestureEvent} fields} *)

  val dollar_gesture_touch_id : touch_id field
  val dollar_gesture_gesture_id : gesture_id field
  val dollar_gesture_num_fingers : int field
  val dollar_gesture_error : float field
  val dollar_gesture_x : float field
  val dollar_gesture_y : float field

  (** {2:drop Drop events}

      {b Warning} If you enable this event {!drop_file_free} must be
      called on the event after you have finished processing it. *)

  val drop_file : event_type
  val drop_file_free : event -> unit


  (** {3 {{:http://wiki.libsdl.org/SDL_DropEvent}SDL_DropEvent}
      fields} *)

  val drop_file_file : event -> string

  (** {2:touch Touch events} *)

  val finger_down : event_type
  val finger_motion : event_type
  val finger_up : event_type

  (** {3 {{:http://wiki.libsdl.org/SDL_TouchFingerEvent}SDL_TouchFingerEvent}
      fields} *)

  val touch_finger_touch_id : touch_id field
  val touch_finger_finger_id : finger_id field
  val touch_finger_x : float field
  val touch_finger_y : float field
  val touch_finger_dx : float field
  val touch_finger_dy : float field
  val touch_finger_pressure : float field

  (** {2:joystickev Joystick events} *)

  val joy_axis_motion : event_type
  val joy_ball_motion : event_type
  val joy_button_down : event_type
  val joy_button_up : event_type
  val joy_device_added : event_type
  val joy_device_removed : event_type
  val joy_hat_motion : event_type

  (** {3 {{:http://wiki.libsdl.org/SDL_JoyAxisEvent}SDL_JoyAxisEvent}
      fields} *)

  val joy_axis_which : joystick_id field
  val joy_axis_axis : uint8 field
  val joy_axis_value : int16 field

  (** {3 {{:http://wiki.libsdl.org/SDL_JoyBallEvent}SDL_JoyBallEvent}
      fields} *)

  val joy_ball_which : joystick_id field
  val joy_ball_ball : uint8 field
  val joy_ball_xrel : int field
  val joy_ball_yrel : int field

  (** {3 {{:http://wiki.libsdl.org/SDL_JoyButtonEvent}SDL_JoyButtonEvent}
      fields} *)

  val joy_button_which : joystick_id field
  val joy_button_button : uint8 field
  val joy_button_state : button_state field

  (** {3 {{:http://wiki.libsdl.org/SDL_JoyDeviceEvent}SDL_JoyDeviceEvent}
      fields} *)

  val joy_device_which : joystick_id field

  (** {3 {{:http://wiki.libsdl.org/SDL_JoyHatEvent}SDL_JoyHatEvent}
      fields} *)

  val joy_hat_which : joystick_id field
  val joy_hat_hat : uint8 field
  val joy_hat_value : Hat.t field

  (** {2:keyboard Keyboard event} *)

  val key_down : event_type
  val key_up : event_type

  (** {3 {{:http://wiki.libsdl.org/SDL_KeyboardEvent}SDL_KeyboardEvent}
      fields} *)

  val keyboard_window_id : int field
  val keyboard_state : button_state field
  val keyboard_repeat : int field
  val keyboard_scancode : scancode field
  val keyboard_keycode : keycode field
  val keyboard_keymod : keymod field

  (** {2:mouse Mouse events} *)

  val mouse_button_down : event_type
  val mouse_button_up : event_type
  val mouse_motion : event_type
  val mouse_wheel : event_type

  (** {3 {{:http://wiki.libsdl.org/SDL_MouseButtonEvent}SDL_MouseButtonEvent}
      fields} *)

  val mouse_button_window_id : int field
  val mouse_button_which : uint32 field
  val mouse_button_button : uint8 field
  val mouse_button_state : button_state field
  val mouse_button_clicks : uint8 field (** SDL 2.0.2 *)
  val mouse_button_x : int field
  val mouse_button_y : int field

  (** {3 {{:http://wiki.libsdl.org/SDL_MouseMotionEvent}SDL_MouseMotionEvent}
      fields} *)

  val mouse_motion_window_id : int field
  val mouse_motion_which : uint32 field
  val mouse_motion_state : uint32 field
  val mouse_motion_x : int field
  val mouse_motion_y : int field
  val mouse_motion_xrel : int field
  val mouse_motion_yrel : int field

  (** {3 {{:http://wiki.libsdl.org/SDL_MouseWheelEvent}SDL_MouseWheelEvent}
      fields} *)

  val mouse_wheel_window_id : int field
  val mouse_wheel_which : uint32 field
  val mouse_wheel_x : int field
  val mouse_wheel_y : int field

  (** {2:multigestureev Multi gesture events} *)

  val multi_gesture : event_type

  (** {3 {{:http://wiki.libsdl.org/SDL_MultiGestureEvent}SDL_MultiGestureEvent}
      fields} *)

  val multi_gesture_touch_id : touch_id field
  val multi_gesture_dtheta : float field
  val multi_gesture_ddist : float field
  val multi_gesture_x : float field
  val multi_gesture_y : float field
  val multi_gesture_num_fingers : int field

  (** {2:quitev Quit events} *)

  val quit : event_type

  (** {2:syswm System window manager events} *)

  val sys_wm_event : event_type

  (** {2:text Text events} *)

  val text_editing : event_type
  val text_input : event_type

  (** {3 {{:http://wiki.libsdl.org/SDL_TextEditingEvent}SDL_TextEditingEvent}
      fields}  *)

  val text_editing_window_id : int field
  val text_editing_text : string field
  val text_editing_start : int field
  val text_editing_length : int field

  (** {3 {{:http://wiki.libsdl.org/SDL_TextInputEvent}SDL_TextInputEvent}
      fields} *)

  val text_input_window_id : int field
  val text_input_text : string field

  (** {2:user User events} *)

  val user_event : event_type

  (** {3 {{:http://wiki.libsdl.org/SDL_UserEvent}SDL_UserEvent} fields} *)

  val user_window_id : int field
  val user_code : int field

  (** {2:window Window events} *)

  val window_event : event_type

  type window_event_id = int
  (** {{:https://wiki.libsdl.org/SDL_WindowEventID}SDL_WindowEventID} *)

  val window_event_enum : window_event_id ->
    [ `Close | `Enter | `Exposed | `Focus_gained | `Focus_lost | `Hidden
    | `Leave | `Maximized | `Minimized | `Moved | `Resized | `Restored
    | `Shown | `Size_changed ]

  val window_event_shown : window_event_id
  val window_event_hidden : window_event_id
  val window_event_exposed : window_event_id
  val window_event_moved : window_event_id
  val window_event_resized : window_event_id
  val window_event_size_changed : window_event_id
  val window_event_minimized : window_event_id
  val window_event_maximized : window_event_id
  val window_event_restored : window_event_id
  val window_event_enter : window_event_id
  val window_event_leave : window_event_id
  val window_event_focus_gained : window_event_id
  val window_event_focus_lost : window_event_id
  val window_event_close : window_event_id

  (** {3 {{:http://wiki.libsdl.org/SDL_WindowEvent}SDL_WindowEvent} fields} *)

  val window_window_id : int field
  val window_event_id : window_event_id field
  val window_data1 : int32 field
  val window_data2 : int32 field

  (** {1:enum Event type enum} *)

  val enum : event_type ->
    [ `App_did_enter_background | `App_did_enter_foreground
    | `App_low_memory | `App_terminating | `App_will_enter_background
    | `App_will_enter_foreground | `Clipboard_update
    | `Controller_axis_motion | `Controller_button_down
    | `Controller_button_up | `Controller_device_added
    | `Controller_device_remapped | `Controller_device_removed
    | `Dollar_gesture | `Dollar_record | `Drop_file | `Finger_down
    | `Finger_motion | `Finger_up | `Joy_axis_motion | `Joy_ball_motion
    | `Joy_button_down | `Joy_button_up | `Joy_device_added
    | `Joy_device_removed | `Joy_hat_motion | `Key_down | `Key_up
    | `Mouse_button_down | `Mouse_button_up | `Mouse_motion
    | `Mouse_wheel | `Multi_gesture | `Quit | `Sys_wm_event
    | `Text_editing | `Text_input | `Unknown | `User_event | `Window_event ]
end

val get_event_state : event_type -> toggle_state
(** {{:http://wiki.libsdl.org/SDL_EventState}SDL_EventState}
    with SDL_QUERY. *)

val set_event_state : event_type -> toggle_state -> unit
(** {{:http://wiki.libsdl.org/SDL_EventState}SDL_EventState}.
    See also {!get_event_state}.  *)

val flush_event : event_type -> unit
(** {{:http://wiki.libsdl.org/SDL_FlushEvent}SDL_FlushEvent} *)

val flush_events : event_type -> event_type -> unit
(** {{:http://wiki.libsdl.org/SDL_FlushEvents}SDL_FlushEvents} *)

val has_event : event_type -> bool
(** {{:http://wiki.libsdl.org/SDL_HasEvent}SDL_HasEvent} *)

val has_events : event_type -> event_type -> bool
(** {{:http://wiki.libsdl.org/SDL_HasEvents}SDL_HasEvents} *)

val poll_event : event option -> bool
(** {{:http://wiki.libsdl.org/SDL_PollEvent}SDL_PollEvent} *)

val pump_events : unit -> unit
(** {{:http://wiki.libsdl.org/SDL_PumpEvents}SDL_PumpEvents} *)

val push_event : event -> bool result
(** {{:http://wiki.libsdl.org/SDL_PushEvent}SDL_PushEvent} *)

val register_event : unit -> event_type option
(** {{:http://wiki.libsdl.org/SDL_RegisterEvents}SDL_RegisterEvents}
    called with [1]. *)

val wait_event : event option -> unit result
(** {{:http://wiki.libsdl.org/SDL_WaitEvent}SDL_WaitEvent} *)

val wait_event_timeout : event option -> int -> bool
(** {{:http://wiki.libsdl.org/SDL_WaitEventTimeout}SDL_WaitEventTimeout} *)

(** {1:forcefeedback
    {{:http://wiki.libsdl.org/CategoryForceFeedback}Force Feedback}} *)

type haptic
type haptic_effect
type haptic_effect_id

module Haptic : sig

  val infinity : uint32

  (** {1 Features} *)

  type feature = int
  val gain : feature
  val autocenter : feature
  val status : feature
  val pause : feature

  (** {1 Directions} *)

  type direction_type = int
  val polar : direction_type
  val cartesian : direction_type
  val spherical : direction_type

  module Direction : sig
    type t
    val create : int -> int32 -> int32 -> int32 -> t
    val typ : t -> direction_type
    val dir_0 : t -> int32
    val dir_1 : t -> int32
    val dir_2 : t -> int32
  end

  (** {1 Effects} *)

  type effect_type = int

  type 'a field
  (** The type for effect fields. *)

  val create_effect : unit -> haptic_effect
  (** [create_effect ()] is an uninitialized haptic effect *)

  val get : haptic_effect -> 'a field -> 'a
  (** [get e f] gets the field f of [e]. *)

  val set : haptic_effect -> 'a field -> 'a -> unit
  (** [set e f v] sets the field f of [e] to [v]. *)

  val typ : effect_type field

  (** {2 Constant effect} *)

  val constant : effect_type

  (** {3 {{:http://wiki.libsdl.org/SDL_HapticConstant}
      SDL_HapticConstant} fields} *)

  val constant_type : effect_type field
  val constant_direction : Direction.t field
  val constant_length : uint32 field
  val constant_delay : uint16 field
  val constant_button : uint16 field
  val constant_interval : uint16 field
  val constant_level : int16 field
  val constant_attack_length : uint16 field
  val constant_attack_level : uint16 field
  val constant_fade_length : uint16 field
  val constant_fade_level : uint16 field

  (** {2 Periodic effect} *)

  val sine : effect_type
  val left_right : effect_type
  val triangle : effect_type
  val sawtooth_up : effect_type
  val sawtooth_down : effect_type

  (** {3 {{:http://wiki.libsdl.org/SDL_HapticPeriodic}
      SDL_HapticPeriodic} fields} *)

  val periodic_type : effect_type field
  val periodic_direction : Direction.t field
  val periodic_length : uint32 field
  val periodic_delay : uint16 field
  val periodic_button : uint16 field
  val periodic_interval : uint16 field
  val periodic_period : uint16 field
  val periodic_magnitude : int16 field
  val periodic_offset : int16 field
  val periodic_phase : uint16 field
  val periodic_attack_length : uint16 field
  val periodic_attack_level : uint16 field
  val periodic_fade_length : uint16 field
  val periodic_fade_level : uint16 field

  (** {2 Condition effect} *)

  val spring : effect_type
  val damper : effect_type
  val inertia : effect_type
  val friction : effect_type

  (** {3 {{:http://wiki.libsdl.org/SDL_HapticCondition}
      SDL_HapticCondition} fields} *)

  val condition_type : effect_type field
  val condition_direction : Direction.t field
  val condition_length : uint32 field
  val condition_delay : uint16 field
  val condition_button : uint16 field
  val condition_interval : uint16 field
  val condition_right_sat_0 : uint16 field
  val condition_right_sat_1 : uint16 field
  val condition_right_sat_2 : uint16 field
  val condition_left_sat_0 : uint16 field
  val condition_left_sat_1 : uint16 field
  val condition_left_sat_2 : uint16 field
  val condition_right_coeff_0 : int16 field
  val condition_right_coeff_1 : int16 field
  val condition_right_coeff_2 : int16 field
  val condition_left_coeff_0 : int16 field
  val condition_left_coeff_1 : int16 field
  val condition_left_coeff_2 : int16 field
  val condition_deadband_0 : uint16 field
  val condition_deadband_1 : uint16 field
  val condition_deadband_2 : uint16 field
  val condition_center_0 : int16 field
  val condition_center_1 : int16 field
  val condition_center_2 : int16 field

  (** {2 Ramp effect} *)

  val ramp : effect_type

  (** {3 {{:http://wiki.libsdl.org/SDL_HapticRamp}SDL_HapticRamp} fields} *)

  val ramp_type : effect_type field
  val ramp_direction : Direction.t field
  val ramp_length : uint32 field
  val ramp_delay : uint16 field
  val ramp_button : uint16 field
  val ramp_interval : uint16 field
  val ramp_start : int16 field
  val ramp_end : int16 field
  val ramp_attack_length : uint16 field
  val ramp_attack_level : uint16 field
  val ramp_fade_length : uint16 field
  val ramp_fade_level : uint16 field

  (** {2 Left right effect}

      For {!left_right}. *)

  (** {3 {{:http://wiki.libsdl.org/SDL_HapticLeftRight}SDL_HapticLeftRight}
      fields} *)

  val left_right_type : effect_type field
  val left_right_length : uint32 field
  val left_right_large_magnitude : uint16 field
  val left_right_small_magnitude : uint16 field

  (** {2 Custom effect} *)

  val custom : effect_type

  (** {3 {{:http://wiki.libsdl.org/SDL_HapticCustom}SDL_HapticCustom} fields} *)

  val custom_type : effect_type field
  val custom_direction : Direction.t field
  val custom_length : uint32 field
  val custom_delay : uint16 field
  val custom_button : uint16 field
  val custom_interval : uint16 field
  val custom_channels : uint8 field
  val custom_period : uint16 field
  val custom_samples : uint16 field
  val custom_data : uint16 list field
  (** {b Note.} Only {!set}able. *)
  val custom_attack_length : uint16 field
  val custom_attack_level : uint16 field
  val custom_fade_length : uint16 field
  val custom_fade_level : uint16 field
end

val haptic_close : haptic -> unit
(** {{:http://wiki.libsdl.org/SDL_HapticClose}SDL_HapticClose} *)

val haptic_destroy_effect : haptic -> haptic_effect_id -> unit
(** {{:http://wiki.libsdl.org/SDL_HapticDestroyEffect}
    SDL_HapticDestroyEffect} *)

val haptic_effect_supported : haptic -> haptic_effect -> bool result
(** {{:http://wiki.libsdl.org/SDL_HapticEffectSupported}
    SDL_HapticEffectSupported} *)

val haptic_get_effect_status : haptic -> haptic_effect_id -> bool result
(** {{:http://wiki.libsdl.org/SDL_HapticGetEffectStatus}
    SDL_HapticGetEffectStatus} *)

val haptic_index : haptic -> int result
(** {{:http://wiki.libsdl.org/SDL_HapticIndex}SDL_HapticIndex} *)

val haptic_name : int -> string result
(** {{:http://wiki.libsdl.org/SDL_HapticName}SDL_HapticName} *)

val haptic_new_effect : haptic -> haptic_effect -> haptic_effect_id result
(** {{:http://wiki.libsdl.org/SDL_HapticNewEffect}SDL_HapticNewEffect} *)

val haptic_num_axes : haptic -> int result
(** {{:http://wiki.libsdl.org/SDL_HapticNumAxes}SDL_HapticNumAxes} *)

val haptic_num_effects : haptic -> int result
(** {{:http://wiki.libsdl.org/SDL_HapticNumEffects}SDL_HapticNumEffects} *)

val haptic_num_effects_playing : haptic -> int result
(** {{:http://wiki.libsdl.org/SDL_HapticNumEffectsPlaying}
    SDL_HapticNumEffectsPlaying} *)

val haptic_open : int -> haptic result
(** {{:http://wiki.libsdl.org/SDL_HapticOpen}SDL_HapticOpen} *)

val haptic_open_from_joystick : joystick -> haptic result
(** {{:http://wiki.libsdl.org/SDL_HapticOpenFromJoystick}
    SDL_HapticOpenFromJoystick} *)

val haptic_open_from_mouse : unit -> haptic result
(** {{:http://wiki.libsdl.org/SDL_HapticOpenFromMouse}
    SDL_HapticOpenFromMouse} *)

val haptic_opened : int -> bool
(** {{:http://wiki.libsdl.org/SDL_HapticOpened}SDL_HapticOpened} *)

val haptic_pause : haptic -> unit result
(** {{:http://wiki.libsdl.org/SDL_HapticPause}SDL_HapticPause} *)

val haptic_query : haptic -> int
(** {{:http://wiki.libsdl.org/SDL_HapticQuery}SDL_HapticQuery} *)

val haptic_rumble_init : haptic -> unit result
(** {{:http://wiki.libsdl.org/SDL_HapticRumbleInit}SDL_HapticRumbleInit} *)

val haptic_rumble_play : haptic -> float -> uint32 -> unit result
(** {{:http://wiki.libsdl.org/SDL_HapticRumblePlay}SDL_HapticRumblePlay} *)

val haptic_rumble_stop : haptic -> unit result
(** {{:http://wiki.libsdl.org/SDL_HapticRumbleStop}SDL_HapticRumbleStop} *)

val haptic_rumble_supported : haptic -> bool result
(** {{:http://wiki.libsdl.org/SDL_HapticRumbleSupported}
    SDL_HapticRumbleSupported} *)

val haptic_run_effect : haptic -> haptic_effect_id -> uint32 ->
  unit result
(** {{:http://wiki.libsdl.org/SDL_HapticRunEffect}SDL_HapticRunEffect} *)

val haptic_set_autocenter : haptic -> int -> unit result
(** {{:http://wiki.libsdl.org/SDL_HapticSetAutocenter}
    SDL_HapticSetAutocenter} *)

val haptic_set_gain : haptic -> int -> unit result
(** {{:http://wiki.libsdl.org/SDL_HapticSetGain}SDL_HapticSetGain} *)

val haptic_stop_all : haptic -> unit result
(** {{:http://wiki.libsdl.org/SDL_HapticStopAll}SDL_HapticStopAll} *)

val haptic_stop_effect : haptic -> haptic_effect_id -> unit result
(** {{:http://wiki.libsdl.org/SDL_HapticStopEffect}SDL_HapticStopEffect} *)

val haptic_unpause : haptic -> unit result
(** {{:http://wiki.libsdl.org/SDL_HapticUnpause}SDL_HapticUnpause} *)

val haptic_update_effect :
  haptic -> haptic_effect_id -> haptic_effect -> unit result
(** {{:http://wiki.libsdl.org/SDL_HapticUpdateEffect}SDL_HapticUpdateEffect} *)

val joystick_is_haptic : joystick -> bool result
(** {{:http://wiki.libsdl.org/SDL_JoystickIsHaptic}SDL_JoystickIsHaptic} *)

val mouse_is_haptic : unit -> bool result
(** {{:http://wiki.libsdl.org/SDL_MouseIsHaptic}SDL_MouseIsHaptic} *)

val num_haptics : unit -> int result
(** {{:http://wiki.libsdl.org/SDL_NumHaptics}SDL_NumHaptics} *)

(** {1:audio {{:http://wiki.libsdl.org/CategoryAudio}Audio}} *)

module Audio : sig

  (** {1:status Audio status} *)

  type status = int
  val stopped : status
  val playing : status
  val paused : status

  (** {1:format Audio format} *)

  type format = int
  (** {{:https://wiki.libsdl.org/SDL_AudioFormat}SDL_AudioFormat} *)

  val s8 : format
  val u8 : format
  val s16_lsb : format
  val s16_msb : format
  val s16_sys : format
  val s16 : format
  val s16_lsb : format
  val u16_lsb : format
  val u16_msb : format
  val u16_sys : format
  val u16 : format
  val u16_lsb : format
  val s32_lsb : format
  val s32_msb : format
  val s32_sys : format
  val s32 : format
  val s32_lsb : format
  val f32_lsb : format
  val f32_msb : format
  val f32_sys : format
  val f32 : format

  (** {1:format Audio allowed changes} *)

  type allow = int
  val allow_frequency_change : int
  val allow_format_change : int
  val allow_channels_change : int
  val allow_any_change : int
end

(** {2:audiodrivers Audio drivers} *)

val audio_init : string option -> unit result
(** {{:http://wiki.libsdl.org/SDL_AudioInit}
    SDL_AudioInit} *)

val audio_quit : unit -> unit
(** {{:http://wiki.libsdl.org/SDL_AudioQuit}
    SDL_AudioQuit} *)

val get_audio_driver : int -> string result
(** {{:http://wiki.libsdl.org/SDL_GetAudioDriver}
    SDL_GetAudioDriver} *)

val get_current_audio_driver : unit -> string option
(** {{:http://wiki.libsdl.org/SDL_GetCurrentAudioDriver}
    SDL_GetCurrentAudioDriver} *)

val get_num_audio_drivers : unit -> int result
(** {{:http://wiki.libsdl.org/SDL_GetNumAudioDrivers}
    SDL_GetNumAudioDrivers} *)

(** {2:audiodevices Audio devices} *)

type audio_device_id

type ('a, 'b) audio_spec =
  { as_freq : int;
    as_format : Audio.format;
    as_channels : uint8;
    as_silence : uint8;
    as_samples : uint8;
    as_size : uint32;
    as_ba_kind : ('a, 'b) Bigarray.kind;
    as_callback : (('a, 'b) bigarray -> unit) option; }
(** {{:http://wiki.libsdl.org/SDL_AudioSpec}SDL_AudioSpec} *)

val close_audio_device : audio_device_id -> unit
(** {{:http://wiki.libsdl.org/SDL_CloseAudioDevice}
    SDL_CloseAudioDevice} *)

val free_wav : ('a, 'b) bigarray -> unit
(** {{:https://wiki.libsdl.org/SDL_FreeWAV}SDL_FreeWAV}. *)

val get_audio_device_name : int -> bool -> string result
(** {{:http://wiki.libsdl.org/SDL_GetAudioDeviceName}
    SDL_GetAudioDeviceName} *)

val get_audio_device_status : audio_device_id -> Audio.status
(** {{:http://wiki.libsdl.org/SDL_GetAudioDeviceStatus}
    SDL_GetAudioDeviceStatus} *)

val get_num_audio_devices : bool -> int result
(** {{:http://wiki.libsdl.org/SDL_GetNumAudioDevices}
    SDL_GetNumAudioDevices} *)

val load_wav_rw : rw_ops -> ('a, 'b) audio_spec ->
  (('a, 'b) audio_spec * ('a, 'b) bigarray) result
(** {{:https://wiki.libsdl.org/SDL_LoadWAV_RW}
    SDL_LoadWAV_RW}. *)

val lock_audio_device : audio_device_id -> unit
(** {{:http://wiki.libsdl.org/SDL_LockAudioDevice}
    SDL_LockAudioDevice} *)

val open_audio_device : string option -> bool -> ('a, 'b) audio_spec ->
  Audio.allow -> (audio_device_id * ('a, 'b) audio_spec) result
(** {{:http://wiki.libsdl.org/SDL_OpenAudioDevice}
    SDL_OpenAudioDevice} *)

val pause_audio_device : audio_device_id -> bool -> unit
(** {{:http://wiki.libsdl.org/SDL_PauseAudioDevice}
    SDL_PauseAudioDevice} *)

val unlock_audio_device : audio_device_id -> unit
(** {{:http://wiki.libsdl.org/SDL_UnlockAudioDevice}
    SDL_UnlockAudioDevice} *)

(*

(** {2:audioconvert Audio conversion} *)

type audio_cvt
(** {{:https://wiki.libsdl.org/SDL_AudioCVT}SDL_AudioCVT} *)

val audio_cvt_mult : audio_cvt -> int * float
(** [audio_cvt_mult cvt] is the [len_mult] and [len_ratio] fields of [cvt] *)

val build_audio_cvt : ~src:Audio.format -> uint8 -> uint8 ~dst:Audio.format ->
  uint8 -> uint8 -> audio_cvt option result
(** {{:http://wiki.libsdl.org/SDL_BuildAudioCVT}
    SDL_BuildAudioCVT}. [None] is returned if no conversion is needed. *)

val convert_audio : audio_cvt -> ('a, 'b) bigarray -> unit
(** {{:http://wiki.libsdl.org/SDL_ConvertAudio}
    SDL_ConvertAudio}. The bigarray has the source and destination *)
*)

(** {1:timer {{:http://wiki.libsdl.org/CategoryTimer}Timer}} *)

val delay : uint32 -> unit
(** {{:http://wiki.libsdl.org/SDL_Delay}SDL_Delay} *)

val get_ticks : unit -> uint32
(** {{:http://wiki.libsdl.org/SDL_GetTicks}SDL_GetTicks} *)

val get_performance_counter : unit -> uint64
(** {{:http://wiki.libsdl.org/SDL_GetPerformanceCounter}
    SDL_GetPerformanceCounter} *)

val get_performance_frequency : unit -> uint64
(** {{:http://wiki.libsdl.org/SDL_GetPerformanceFrequency}
    SDL_GetPerformanceFrequency} *)

(** {1:platform Platform and CPU information} *)

val get_platform : unit -> string
(** {{:http://wiki.libsdl.org/SDL_GetPlatform}SDL_GetPlatform} *)

val get_cpu_cache_line_size : unit -> int result
(** {{:http://wiki.libsdl.org/SDL_GetCPUCacheLineSize}
    SDL_GetCPUCacheLineSize} *)

val get_cpu_count : unit -> int
(** {{:http://wiki.libsdl.org/SDL_GetCPUCount}SDL_GetCPUCount} *)

val get_system_ram : unit -> int
(** {{:http://wiki.libsdl.org/SDL_GetSystemRAM}SDL_GetSystemRAM} *)

val has_3d_now : unit -> bool
(** {{:http://wiki.libsdl.org/SDL_Has3DNow}SDL_Has3DNow} *)

val has_altivec : unit -> bool
(** {{:http://wiki.libsdl.org/SDL_HasAltiVec}SDL_HasAltiVec} *)

val has_avx : unit -> bool
(** {{:https://wiki.libsdl.org/SDL_HasAVX}SDL_HasAVX} (SDL 2.0.2) *)

val has_mmx : unit -> bool
(** {{:http://wiki.libsdl.org/SDL_HasMMX}SDL_HasMMX} *)

val has_rdtsc : unit -> bool
(** {{:http://wiki.libsdl.org/SDL_HasRDTSC}SDL_HasRDTSC} *)

val has_sse : unit -> bool
(** {{:http://wiki.libsdl.org/SDL_HasSSE}SDL_HasSSE} *)

val has_sse2 : unit -> bool
(** {{:http://wiki.libsdl.org/SDL_HasSSE2}SDL_HasSSE2} *)

val has_sse3 : unit -> bool
(** {{:http://wiki.libsdl.org/SDL_HasSSE3}SDL_HasSSE3} *)

val has_sse41 : unit -> bool
(** {{:http://wiki.libsdl.org/SDL_HasSSE3}SDL_HasSSE41} *)

val has_sse42 : unit -> bool
(** {{:http://wiki.libsdl.org/SDL_HasSSE3}SDL_HasSSE42} *)

(** {1:power {{:http://wiki.libsdl.org/CategoryPower}Power}} *)

type power_state =
  [ `Unknown | `On_battery | `No_battery | `Charging | `Charged ]
(** {{:http://wiki.libsdl.org/SDL_PowerState}SDL_PowerState} *)

type power_info =
  { pi_state : power_state;
    pi_secs : int option;
    pi_pct : int option; }

val get_power_info : unit -> power_info
(** {{:http://wiki.libsdl.org/SDL_GetPowerInfo}SDL_GetPowerInfo} *)

(**     {1:coverage Binding Coverage}

    Everything except the following functions/categories are available.

    {2 Unbound categories}

    {ul
    {- {{:http://wiki.libsdl.org/CategoryAssertions}Assertions}
        (cpp based).}
    {- {{:https://wiki.libsdl.org/CategorySWM}Platform-specific Window
        Management} (not useful at the moment)}
    {- {{:http://wiki.libsdl.org/CategoryThread}Thread Management}
        (better use another OCaml API)}
    {- {{:http://wiki.libsdl.org/CategoryMutex}Thread Synchronization
        Primitives} (better use another OCaml API)}
    {- {{:http://wiki.libsdl.org/CategoryAtomic}Atomic Operations}
        (mostly cpp based)}
    {- {{:http://wiki.libsdl.org/CategoryIO}File I/O Abstraction}
        (only the minimum was covered for other parts of the API that needs
        it, better use another OCaml API)}
    {- {{:http://wiki.libsdl.org/CategorySharedObject}
       Shared Object Loading and Function Lookup} (use ocaml-ctypes)}
    {- {{:http://wiki.libsdl.org/CategoryEndian}Byte Order and Byte Swapping}
        (cpp based)}
    {- {{:http://wiki.libsdl.org/CategoryBits}Bit Manipulation}
        (cpp based)}}

    {2 Unbound functions}

    {ul
    {- {{:https://wiki.libsdl.org/SDL_AddHintCallback}SDL_AddHintCallback}
        (avoid callbacks from C to OCaml)}
    {- {{:https://wiki.libsdl.org/SDL_DelHintCallback}SDL_DelHintCallback}
        (avoid callbacks from C to OCaml)}
    {- {{:http://wiki.libsdl.org/SDL_LogGetOutputFunction}
       SDL_LogGetOutputFunction} (avoid callbacks from C to OCaml)}
    {- {{:http://wiki.libsdl.org/SDL_LogSetOutputFunction}
       SDL_LogSetOutputFunction} (avoid callbacks from C to OCaml)}
    {- {{:http://wiki.libsdl.org/SDL_CreateWindowFrom}SDL_CreateWindowFrom}
        (avoid [void *] type in the interface)}
    {- {{:http://wiki.libsdl.org/SDL_GetWindowData}SDL_GetWindowData}
        (avoid storing OCaml values in C)}
    {- {{:http://wiki.libsdl.org/SDL_SetWindowData}SDL_SetWindowData}
        (avoid storing OCaml values in C)}
    {- {{:http://wiki.libsdl.org/SDL_GetWindowWMInfo}SDL_GetWindowWMInfo}
        (avoid [void *] type in the interface)}
    {- {{:http://wiki.libsdl.org/SDL_GL_GetProcAddress}SDL_GL_GetProcAddress}
        (use another OCaml API)}
    {- {{:http://wiki.libsdl.org/SDL_GL_LoadLibrary}SDL_GL_LoadLibrary}
        (use another OCaml API)}
    {- {{:http://wiki.libsdl.org/SDL_GL_UnloadLibrary}SDL_GL_UnloadLibrary}
        (use another OCaml API)}
    {- {{:http://wiki.libsdl.org/SDL_AddEventWatch}SDL_AddEventWatch}
        (avoid callbacks from C to OCaml)}
    {- {{:http://wiki.libsdl.org/SDL_DelEventWatch}SDL_DelEventWatch}
        (avoid callbacks from C to OCaml)}
    {- {{:http://wiki.libsdl.org/SDL_FilterEvents}SDL_FilterEvents}
        (avoid callbacks from C to OCaml)}
    {- {{:http://wiki.libsdl.org/SDL_GetEventFilter}SDL_GetEventFilter}
        (avoid callbacks from C to OCaml)}
    {- {{:http://wiki.libsdl.org/SDL_SetEventFilter}SDL_SetEventFilter}
        (avoid callbacks from C to OCaml)}
    {- {{:http://wiki.libsdl.org/SDL_PeepEvents}SDL_PeepEvents}
        (Should certainly be split into more than one fun,
        functionality also available through other bound functions.)}
    {- {{:http://wiki.libsdl.org/SDL_QuitRequested}SDL_QuitRequested}
        (cpp based)}
    {- {{:http://wiki.libsdl.org/SDL_AddTimer}SDL_AddTimer}
        (avoid callbacks from C to OCaml, besides callbacks are
        run on another thread, thus runtime lock support in ocaml-ctypes
        is needed. Probably better to use another OCaml API anyway)}
    {- {{:http://wiki.libsdl.org/SDL_RemoveTimer}SDL_RemoveTimer}
        (avoid callbacks from C to OCaml)}
    {- {{:http://wiki.libsdl.org/SDL_GetAudioStatus}SDL_GetAudioStatus}
        (SDL legacy function)}
    {- {{:http://wiki.libsdl.org/SDL_OpenAudio}SDL_OpenAudio}
        (SDL legacy function)}
    {- {{:http://wiki.libsdl.org/SDL_CloseAudio}SDL_CloseAudio}
        (SDL legacy function)}
    {- {{:http://wiki.libsdl.org/SDL_LockAudio}SDL_LockAudio}
        (SDL legacy function)}
    {- {{:http://wiki.libsdl.org/SDL_MixAudio}SDL_MixAudio}
        (SDL legacy function)}
    {- {{:http://wiki.libsdl.org/SDL_MixAudioFormat}
        SDL_MixAudioFormat} (limited functionality, do your own mixing).}
    {- {{:http://wiki.libsdl.org/SDL_PauseAudio}SDL_PauseAudio}
        (SDL legacy function)}
    {- {{:http://wiki.libsdl.org/SDL_UnlockAudio}SDL_UnlockAudio}
        (SDL legacy function)}} *)
end

(** {1:conventions Binding conventions}
    {2:naming Naming}

    C names are transformed as follows. The [SDL_] is mapped to the
    module name {!Sdl}, for the rest add an underscore between each
    minuscule and majuscule and lower case the result
    (e.g. [SDL_GetError] maps to {!Sdl.get_error}). Part of the name
    may also be wrapped by a module, (e.g. SDL_INIT_VIDEO becomes
    {!Sdl.Init.video}). If you open {!Tsdl}, your code will look
    mostly like SDL code but in accordance with OCaml's programming
    conventions. Exceptions to the naming convention do occur for
    technical reasons.

    {2:errors Errors}

    All functions that return an {!Sdl.result} have the string
    returned by [Sdl.get_error ()] in the [Error (`Msg _)] case.

    {2:enums Bit fields and enumerants}

    Most bit fields and enumerants are not mapped to variants, they
    are represented by OCaml values of a given abstract type in a
    specific module with a composition operator to combine them and a
    testing operator to test them. The flags for initializing SDL in the
    module {!Sdl.Init} is an example of that:
{[
match Sdl.init Sdl.Init.(video + timer + audio) with
| Error _ -> ...
| Ok () -> ...
]}
    Using variants in that case is inconvenient for the binding
    function and of limited use since most of the time bit fields are
    given to setup state and, as such, are less likley to be used for
    pattern matching. *)

(** {1:examples Examples}

    {2:toplevel Toplevel}

To use [Tsdl] in the toplevel with [findlib] just issue:
{[
> #use "topfind";;
> #require "tsdl.top";;
]}

This automatically loads the library and opens the [Tsdl] module.

    {2:opengl OpenGL window}

    The following is the minimum you need to get a working OpenGL window
    with SDL.
{[
open Tsdl
open Result

let main () = match Sdl.init Sdl.Init.video with
| Error (`Msg e) -> Sdl.log "Init error: %s" e; exit 1
| Ok () ->
    match Sdl.create_window ~w:640 ~h:480 "SDL OpenGL" Sdl.Window.opengl with
    | Error (`Msg e) -> Sdl.log "Create window error: %s" e; exit 1
    | Ok w ->
        Sdl.delay 3000l;
        Sdl.destroy_window w;
        Sdl.quit ();
        exit 0

let () = main ()
]}

This can be compiled to byte and native code with:
{v
> ocamlfind ocamlc -package tsdl -linkpkg -o min.byte min.ml
> ocamlfind ocamlopt -package tsdl -linkpkg -o min.native min.ml
v}

*)

(*---------------------------------------------------------------------------
   Copyright (c) 2013 Daniel C. Bünzli

   Permission to use, copy, modify, and/or distribute this software for any
   purpose with or without fee is hereby granted, provided that the above
   copyright notice and this permission notice appear in all copies.

   THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
   WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
   MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
   ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
   WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
   ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
   OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
  ---------------------------------------------------------------------------*)

end = struct
#1 "tsdl.ml"
(*---------------------------------------------------------------------------
   Copyright (c) 2013 Daniel C. Bünzli. All rights reserved.
   Distributed under the ISC license, see terms at the end of the file.
   tsdl v0.9.1
  ---------------------------------------------------------------------------*)

let unsafe_get = Array.unsafe_get

open Ctypes
open Foreign

module Sdl = struct

(* Enum cases and #ifdef'd constants, see support/ in the distribution *)

open Tsdl_consts

(* Formatting with continuation. *)

let kpp k fmt =
  let k fmt = k (Format.flush_str_formatter ()) in
  Format.kfprintf k Format.str_formatter fmt

(* Invalid_argument strings *)

let str = Printf.sprintf
let err_index i = str "invalid index: %d" i
let err_length_mul l mul = str "invalid length: %d not a multiple of %d" l mul
let err_drop_file = "null file name (drop_file_free already called ?)"
let err_read_field = "cannot read field"
let err_bigarray_pitch pitch ba_el_size =
  str "invalid bigarray kind: pitch (%d bytes) not a multiple of bigarray \
       element byte size (%d)" pitch ba_el_size

let err_bigarray_data len ba_el_size =
  str "invalid bigarray kind: data (%d bytes) not a multiple of bigarray \
       element byte size (%d)" len ba_el_size

(* ctypes views *)

let write_never _ = assert false

let bool =
  view ~read:((<>)0) ~write:(fun b -> compare b false) int;;

let int_as_uint8_t =
  view ~read:Unsigned.UInt8.to_int ~write:Unsigned.UInt8.of_int uint8_t

let int_as_uint16_t =
  view ~read:Unsigned.UInt16.to_int ~write:Unsigned.UInt16.of_int uint16_t

let int_as_uint32_t =
  view ~read:Unsigned.UInt32.to_int ~write:Unsigned.UInt32.of_int uint32_t

let int_as_int32_t =
  view ~read:Signed.Int32.to_int ~write:Signed.Int32.of_int int32_t

let int32_as_uint32_t =
  view ~read:Unsigned.UInt32.to_int32 ~write:Unsigned.UInt32.of_int32 uint32_t

let string_as_char_array n = (* FIXME: drop this if ctypes proposes better *)
  let n_array = array n char in
  let read a =
    let len = CArray.length a in
    let b = Buffer.create len in
    try
      for i = 0 to len - 1 do
        let c = CArray.get a i in
        if c = '\000' then raise Exit else Buffer.add_char b c
      done;
      Buffer.contents b
    with Exit -> Buffer.contents b
  in
  let write s =
    let a = CArray.make char n in
    let len = min (CArray.length a) (String.length s) in
    for i = 0 to len - 1 do CArray.set a i (s.[i]) done;
    a
  in
  view ~read ~write n_array

let get_error =
  foreign "SDL_GetError" (void @-> returning string)

(* SDL results *)
open Result
type 'a result = ( 'a, [ `Msg of string ] ) Result.result

let error () = Error (`Msg (get_error ()))

let zero_to_ok =
  let read = function 0 -> Ok () | err -> error () in
  view ~read ~write:write_never int

let one_to_ok =
  let read = function 1 -> Ok () | err -> error () in
  view ~read ~write:write_never int

let bool_to_ok =
  let read = function 0 -> Ok false | 1 -> Ok true | _ -> error () in
  view ~read ~write:write_never int

let nat_to_ok =
  let read = function n when n < 0 -> error () | n -> Ok n in
  view ~read ~write:write_never int

let some_to_ok t =
  let read = function Some v -> Ok v | None -> error () in
  view ~read ~write:write_never t

let sdl_free = foreign "SDL_free" (ptr void @-> returning void)

(* Since we never let SDL redefine our main make sure this is always
   called. *)

let () =
  let set_main_ready = foreign "SDL_SetMainReady" (void @-> returning void) in
  set_main_ready ()

let stub = true


(* Integer types and maps *)

type uint8 = int
type uint16 = int
type int16 = int
type uint32 = int32
type uint64 = int64

module Int = struct type t = int let compare : int -> int -> int = compare end
module Imap = Map.Make(Int)

(* Bigarrays *)

type ('a, 'b) bigarray = ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t

let ba_create k len = Bigarray.Array1.create k Bigarray.c_layout len
let ba_kind_byte_size :  ('a, 'b) Bigarray.kind -> int = fun k ->
  let open Bigarray in
  (* FIXME: see http://caml.inria.fr/mantis/view.php?id=6263 *)
  match Obj.magic k with
  | k when k = char || k = int8_signed || k = int8_unsigned -> 1
  | k when k = int16_signed || k = int16_unsigned -> 2
  | k when k = int32 || k = float32 -> 4
  | k when k = float64 || k = int64 || k = complex32 -> 8
  | k when k = complex64 -> 16
  | k when k = int || k = nativeint -> Sys.word_size / 8
  | k -> assert false

let access_ptr_typ_of_ba_kind : ('a, 'b) Bigarray.kind -> 'a ptr typ = fun k ->
  let open Bigarray in
  (* FIXME: use typ_of_bigarray_kind when ctypes support it. *)
  match Obj.magic k with
  | k when k = float32 -> Obj.magic (ptr Ctypes.float)
  | k when k = float64 -> Obj.magic (ptr Ctypes.double)
  | k when k = complex32 -> Obj.magic (ptr Ctypes.complex32)
  | k when k = complex64 -> Obj.magic (ptr Ctypes.complex64)
  | k when k = int8_signed -> Obj.magic (ptr Ctypes.int8_t)
  | k when k = int8_unsigned -> Obj.magic (ptr Ctypes.uint8_t)
  | k when k = int16_signed -> Obj.magic (ptr Ctypes.int16_t)
  | k when k = int16_unsigned -> Obj.magic (ptr Ctypes.uint16_t)
  | k when k = int -> Obj.magic (ptr Ctypes.camlint)
  | k when k = int32 -> Obj.magic (ptr Ctypes.int32_t)
  | k when k = int64 -> Obj.magic (ptr Ctypes.int64_t)
  | k when k = nativeint -> Obj.magic (ptr Ctypes.nativeint)
  | k when k = char -> Obj.magic (ptr Ctypes.char)
  | _ -> assert false

let ba_byte_size ba =
  let el_size = ba_kind_byte_size (Bigarray.Array1.kind ba) in
  el_size * Bigarray.Array1.dim ba

(* Basics *)

(* Initialization and shutdown *)

module Init = struct
  type t = Unsigned.uint32
  let i = Unsigned.UInt32.of_int
  let ( + ) = Unsigned.UInt32.logor
  let test f m = Unsigned.UInt32.(compare (logand f m) zero <> 0)
  let eq f f' = Unsigned.UInt32.(compare f f' = 0)
  let timer = i sdl_init_timer
  let audio = i sdl_init_audio
  let video = i sdl_init_video
  let joystick = i sdl_init_joystick
  let haptic = i sdl_init_haptic
  let gamecontroller = i sdl_init_gamecontroller
  let events = i sdl_init_events
  let everything = i sdl_init_everything
  let noparachute = i sdl_init_noparachute
end

let init =
  foreign "SDL_Init" (uint32_t @-> returning zero_to_ok)

let init_sub_system =
  foreign "SDL_InitSubSystem" (uint32_t @-> returning zero_to_ok)

let quit =
  foreign "SDL_Quit" (void @-> returning void)

let quit_sub_system =
  foreign "SDL_QuitSubSystem" (uint32_t @-> returning void)

let was_init =
  foreign "SDL_WasInit" (uint32_t @-> returning uint32_t)

let was_init = function
| None -> was_init (Unsigned.UInt32.of_int 0)
| Some m -> was_init m

(* Hints *)

module Hint = struct
  type t = string
  let framebuffer_acceleration = sdl_hint_framebuffer_acceleration
  let idle_timer_disabled = sdl_hint_idle_timer_disabled
  let orientations = sdl_hint_orientations
  let render_driver = sdl_hint_render_driver
  let render_opengl_shaders = sdl_hint_render_opengl_shaders
  let render_scale_quality = sdl_hint_render_scale_quality
  let render_vsync = sdl_hint_render_vsync

  type priority = int
  let default = sdl_hint_default
  let normal = sdl_hint_normal
  let override = sdl_hint_override
end

let clear_hints =
  foreign "SDL_ClearHints" (void @-> returning void)

let get_hint =
  foreign "SDL_GetHint" (string @-> returning string_opt)

let set_hint =
  foreign "SDL_SetHint" (string @-> string @-> returning bool)

let set_hint_with_priority =
  foreign "SDL_SetHintWithPriority"
    (string @-> string @-> int @-> returning bool)

(* Errors *)

let clear_error =
  foreign "SDL_ClearError" (void @-> returning void)

let set_error =
  foreign "SDL_SetError" (string @-> returning int)

let set_error fmt =
  kpp (fun s -> ignore (set_error s)) fmt

(* Log *)

module Log = struct
  type category = int
  let category_application = sdl_log_category_application
  let category_error = sdl_log_category_error
  let category_system = sdl_log_category_system
  let category_audio = sdl_log_category_audio
  let category_video = sdl_log_category_video
  let category_render = sdl_log_category_render
  let category_input = sdl_log_category_input
  let category_custom = sdl_log_category_custom

  type priority = int
  let priority_compare : int -> int -> int = Pervasives.compare
  let priority_verbose = sdl_log_priority_verbose
  let priority_debug = sdl_log_priority_debug
  let priority_info = sdl_log_priority_info
  let priority_warn = sdl_log_priority_warn
  let priority_error = sdl_log_priority_error
  let priority_critical = sdl_log_priority_critical
end

let log_fun_t = (int @-> string @-> string @-> returning void)

let log =
  foreign "SDL_Log" (string @-> string @-> returning void)

let log fmt =
  kpp (fun s -> ignore (log "%s" s)) fmt

let log_critical =
  foreign "SDL_LogCritical" log_fun_t

let log_critical c fmt =
  kpp (fun s -> ignore (log_critical c "%s" s)) fmt

let log_debug =
  foreign "SDL_LogDebug" log_fun_t

let log_debug c fmt =
  kpp (fun s -> ignore (log_debug c "%s" s)) fmt

let log_error =
  foreign "SDL_LogError" log_fun_t

let log_error c fmt =
  kpp (fun s -> ignore (log_error c "%s" s)) fmt

let log_info =
  foreign "SDL_LogInfo" log_fun_t

let log_info c fmt =
  kpp (fun s -> ignore (log_info c "%s" s)) fmt

let log_verbose =
  foreign "SDL_LogVerbose" log_fun_t

let log_verbose c fmt =
  kpp (fun s -> ignore (log_verbose c "%s" s)) fmt

let log_warn =
  foreign "SDL_LogWarn" log_fun_t

let log_warn c fmt =
  kpp (fun s -> ignore (log_warn c "%s" s)) fmt

let log_get_priority =
  foreign "SDL_LogGetPriority" (int @-> returning int)

let log_message =
  foreign "SDL_LogMessage"
    (int @-> int @-> string @-> string @-> returning void)

let log_message c p fmt =
  kpp (fun s -> ignore (log_message c p "%s" s)) fmt

let log_reset_priorities =
  foreign "SDL_LogResetPriorities" (void @-> returning void)

let log_set_all_priority =
  foreign "SDL_LogSetAllPriority" (int @-> returning void)

let log_set_priority =
  foreign "SDL_LogSetPriority" (int @-> int @-> returning void)

(* Version *)

let version = structure "SDL_version"
let version_major = field version "major" uint8_t
let version_minor = field version "minor" uint8_t
let version_patch = field version "patch" uint8_t
let () = seal version

let get_version =
  foreign "SDL_GetVersion" (ptr version @-> returning void)

let get_version () =
  let get v f = Unsigned.UInt8.to_int (getf v f) in
  let v = make version in
  get_version (addr v);
  (get v version_major), (get v version_minor), (get v version_patch)

let get_revision =
  foreign "SDL_GetRevision" (void @-> returning string)

let get_revision_number =
  foreign "SDL_GetRevisionNumber" (void @-> returning int)

(* IO absraction *)

type _rw_ops
let rw_ops_struct : _rw_ops structure typ = structure "SDL_RWops"
let rw_ops : _rw_ops structure ptr typ = ptr rw_ops_struct
let rw_ops_opt : _rw_ops structure ptr option typ = ptr_opt rw_ops_struct

let rw_ops_size = field rw_ops_struct "size"
    (funptr (rw_ops @-> returning int64_t))
let rw_ops_seek = field rw_ops_struct "seek"
    (funptr (rw_ops @-> int64_t @-> int @-> returning int64_t))
let rw_ops_read = field rw_ops_struct "read"
    (funptr (rw_ops @-> ptr void @-> size_t @-> size_t @-> returning size_t))
let rw_ops_write = field rw_ops_struct "write"
    (funptr (rw_ops @-> ptr void @-> size_t @-> size_t @-> returning size_t))
let rw_ops_close = field rw_ops_struct "close"
    (funptr (rw_ops @-> returning int))
let _ = field rw_ops_struct "type" uint32_t
(* ... #ifdef'd union follows, we don't care we don't use Ctypes.make *)
let () = seal rw_ops_struct

type rw_ops = _rw_ops structure ptr

let rw_from_file =
  foreign "SDL_RWFromFile"
    (string @-> string @-> returning (some_to_ok rw_ops_opt))

let rw_close ops =
  let close = getf (!@ ops) rw_ops_close in
  if close ops = 0 then Ok () else (error ())

let unsafe_rw_ops_of_ptr addr : rw_ops =
  from_voidp rw_ops_struct (ptr_of_raw_address addr)
let unsafe_ptr_of_rw_ops rw_ops =
  raw_address_of_ptr (to_voidp rw_ops)

(* File system paths *)

let get_base_path =
  foreign "SDL_GetBasePath" (void @-> returning (ptr char))

let get_base_path () =
  let p = get_base_path () in
  let path = coerce (ptr char) (some_to_ok string_opt) p in
  sdl_free (coerce (ptr char) (ptr void) p);
  path

let get_pref_path =
  foreign "SDL_GetPrefPath" (string @-> string @-> returning (ptr char))

let get_pref_path ~org ~app =
  let p = get_pref_path org app in
  let path = coerce (ptr char) (some_to_ok string_opt) p in
  sdl_free (coerce (ptr char) (ptr void) p);
  path

(* Video *)

type window = unit ptr
let window : window typ = ptr void
let window_opt : window option typ = ptr_opt void

let unsafe_window_of_ptr addr : window =
  ptr_of_raw_address addr
let unsafe_ptr_of_window window =
  raw_address_of_ptr (to_voidp window)

(* Colors *)

type _color
type color = _color structure
let color : color typ = structure "SDL_Color"
let color_r = field color "r" uint8_t
let color_g = field color "g" uint8_t
let color_b = field color "b" uint8_t
let color_a = field color "a" uint8_t
let () = seal color

module Color = struct
  let create ~r ~g ~b ~a =
    let c = make color in
    setf c color_r (Unsigned.UInt8.of_int r);
    setf c color_g (Unsigned.UInt8.of_int g);
    setf c color_b (Unsigned.UInt8.of_int b);
    setf c color_a (Unsigned.UInt8.of_int a);
    c

  let r c = Unsigned.UInt8.to_int (getf c color_r)
  let g c = Unsigned.UInt8.to_int (getf c color_g)
  let b c = Unsigned.UInt8.to_int (getf c color_b)
  let a c = Unsigned.UInt8.to_int (getf c color_a)

  let set_r c r = setf c color_r (Unsigned.UInt8.of_int r)
  let set_g c g = setf c color_g (Unsigned.UInt8.of_int g)
  let set_b c b = setf c color_b (Unsigned.UInt8.of_int b)
  let set_a c a = setf c color_a (Unsigned.UInt8.of_int a)
end

(* Points *)

type _point
type point = _point structure
let point : point typ = structure "SDL_Point"
let point_x = field point "x" int
let point_y = field point "y" int
let () = seal point

module Point = struct
  let create ~x ~y =
    let p = make point in
    setf p point_x x;
    setf p point_y y;
    p

  let x p = getf p point_x
  let y p = getf p point_y

  let set_x p x = setf p point_x x
  let set_y p y = setf p point_y y

  let opt_addr = function
  | None -> coerce (ptr void) (ptr point) null
  | Some v -> addr v
end

(* Rectangle *)

type _rect
type rect = _rect structure
let rect : rect typ = structure "SDL_Rect"
let rect_x = field rect "x" int
let rect_y = field rect "y" int
let rect_w = field rect "w" int
let rect_h = field rect "h" int
let () = seal rect

module Rect = struct
  let create ~x ~y ~w ~h =
    let r = make rect in
    setf r rect_x x;
    setf r rect_y y;
    setf r rect_w w;
    setf r rect_h h;
    r

  let x r = getf r rect_x
  let y r = getf r rect_y
  let w r = getf r rect_w
  let h r = getf r rect_h

  let set_x r x = setf r rect_x x
  let set_y r y = setf r rect_y y
  let set_w r w = setf r rect_w w
  let set_h r h = setf r rect_h h

  let opt_addr = function
  | None -> coerce (ptr void) (ptr rect) null
  | Some v -> addr v
end

let enclose_points =
  foreign "SDL_EnclosePoints"
    (ptr void @-> int @-> ptr rect @-> ptr rect @-> returning bool)

let enclose_points_ba ?clip ps =
  let len = Bigarray.Array1.dim ps in
  if len mod 2 <> 0 then invalid_arg (err_length_mul len 2) else
  let count = len / 2 in
  let ps = to_voidp (bigarray_start array1 ps) in
  let res = make rect in
  if enclose_points ps count (Rect.opt_addr clip) (addr res)
  then Some res
  else None

let enclose_points ?clip ps =
  let count = List.length ps in
  let ps = to_voidp (CArray.start (CArray.of_list point ps)) in
  let res = make rect in
  if enclose_points ps count (Rect.opt_addr clip) (addr res)
  then Some res
  else None

let has_intersection =
  foreign "SDL_HasIntersection"
    (ptr rect @-> ptr rect @-> returning bool)

let has_intersection a b =
  has_intersection (addr a) (addr b)

let intersect_rect =
  foreign "SDL_IntersectRect"
    (ptr rect @-> ptr rect @-> ptr rect @-> returning bool)

let intersect_rect a b =
  let res = make rect in
  if intersect_rect (addr a) (addr b) (addr res) then Some res else None

let intersect_rect_and_line =
  foreign "SDL_IntersectRectAndLine"
    (ptr rect @-> ptr int @-> ptr int @-> ptr int @-> ptr int @->
     returning bool)

let intersect_rect_and_line r x1 y1 x2 y2 =
  let alloc v = allocate int v in
  let x1, y1 = alloc x1, alloc y1 in
  let x2, y2 = alloc x2, alloc y2 in
  if intersect_rect_and_line (addr r) x1 y1 x2 y2
  then Some ((!@x1, !@y1), (!@x2, !@y2))
  else None

let rect_empty r =
  (* symbol doesn't exist: SDL_FORCE_INLINE directive
     foreign "SDL_RectEmpty" (ptr rect @-> returning bool) *)
  Rect.w r <= 0 || Rect.h r <= 0

let rect_equals a b =
  (* symbol doesn't exist: SDL_FORCE_INLINE directive
    foreign "SDL_RectEquals" (ptr rect @-> ptr rect @-> returning bool) *)
  (Rect.x a = Rect.x b) && (Rect.y a = Rect.y b) &&
  (Rect.w a = Rect.w b) && (Rect.h a = Rect.h b)

let union_rect =
  foreign "SDL_UnionRect"
    (ptr rect @-> ptr rect @-> ptr rect @-> returning void)

let union_rect a b =
  let res = make rect in
  union_rect (addr a) (addr b) (addr res);
  res

(* Palettes *)

type _palette
type palette_struct = _palette structure
let palette_struct : palette_struct typ = structure "SDL_Palette"
let palette_ncolors = field palette_struct "ncolors" int
let palette_colors = field palette_struct "colors" (ptr color)
let _ = field palette_struct "version" uint32_t
let _ = field palette_struct "refcount" int
let () = seal palette_struct

type palette = palette_struct ptr
let palette : palette typ = ptr palette_struct
let palette_opt : palette option typ = ptr_opt palette_struct

let unsafe_palette_of_ptr addr : palette =
  from_voidp palette_struct (ptr_of_raw_address addr)
let unsafe_ptr_of_palette palette =
  raw_address_of_ptr (to_voidp palette)

let alloc_palette =
  foreign "SDL_AllocPalette"
    (int @-> returning (some_to_ok palette_opt))

let free_palette =
  foreign "SDL_FreePalette" (palette @-> returning void)

let get_palette_ncolors p =
  getf (!@ p) palette_ncolors

let get_palette_colors p =
  let ps = !@ p in
  CArray.to_list
    (CArray.from_ptr (getf ps palette_colors) (getf ps palette_ncolors))

let get_palette_colors_ba p =
  let ps = !@ p in
  (* FIXME: ctypes should have a CArray.copy function *)
  let n = getf ps palette_ncolors in
  let ba = Bigarray.(Array1.create int8_unsigned c_layout (n * 4)) in
  let ba_ptr =
    CArray.from_ptr (coerce (ptr int) (ptr color) (bigarray_start array1 ba)) n
  in
  let ca = CArray.from_ptr (getf ps palette_colors) n in
  for i = 0 to n - 1 do CArray.set ba_ptr i (CArray.get ca i) done;
  ba

let set_palette_colors =
  foreign "SDL_SetPaletteColors"
    (palette @-> ptr void @-> int @-> int @-> returning zero_to_ok)

let set_palette_colors_ba p cs ~fst =
  let len = Bigarray.Array1.dim cs in
  if len mod 4 <> 0 then invalid_arg (err_length_mul len 4) else
  let count = len / 4 in
  let cs = to_voidp (bigarray_start array1 cs) in
  set_palette_colors p cs fst count

let set_palette_colors p cs ~fst =
  let count = List.length cs in
  let a = CArray.of_list color cs in
  set_palette_colors p (to_voidp (CArray.start a)) fst count

(* Pixel formats *)

type gamma_ramp = (int, Bigarray.int16_unsigned_elt) bigarray

let calculate_gamma_ramp =
  foreign "SDL_CalculateGammaRamp"
    (float @-> ptr void @-> returning void)

let calculate_gamma_ramp g =
  let ba = Bigarray.(Array1.create int16_unsigned c_layout 256) in
  calculate_gamma_ramp g (to_voidp (bigarray_start array1 ba));
  ba

module Blend = struct
  type mode = int
  let mode_none = sdl_blendmode_none
  let mode_blend = sdl_blendmode_blend
  let mode_add = sdl_blendmode_add
  let mode_mod = sdl_blendmode_mod
end

module Pixel = struct
  type format_enum = Unsigned.UInt32.t
  let i = Unsigned.UInt32.of_int32
  let to_uint32 = Unsigned.UInt32.to_int32
  let eq f f' = Unsigned.UInt32.(compare f f' = 0)
  let format_unknown = i sdl_pixelformat_unknown
  let format_index1lsb = i sdl_pixelformat_index1lsb
  let format_index1msb = i sdl_pixelformat_index1msb
  let format_index4lsb = i sdl_pixelformat_index4lsb
  let format_index4msb = i sdl_pixelformat_index4msb
  let format_index8 = i sdl_pixelformat_index8
  let format_rgb332 = i sdl_pixelformat_rgb332
  let format_rgb444 = i sdl_pixelformat_rgb444
  let format_rgb555 = i sdl_pixelformat_rgb555
  let format_bgr555 = i sdl_pixelformat_bgr555
  let format_argb4444 = i sdl_pixelformat_argb4444
  let format_rgba4444 = i sdl_pixelformat_rgba4444
  let format_abgr4444 = i sdl_pixelformat_abgr4444
  let format_bgra4444 = i sdl_pixelformat_bgra4444
  let format_argb1555 = i sdl_pixelformat_argb1555
  let format_rgba5551 = i sdl_pixelformat_rgba5551
  let format_abgr1555 = i sdl_pixelformat_abgr1555
  let format_bgra5551 = i sdl_pixelformat_bgra5551
  let format_rgb565 = i sdl_pixelformat_rgb565
  let format_bgr565 = i sdl_pixelformat_bgr565
  let format_rgb24 = i sdl_pixelformat_rgb24
  let format_bgr24 = i sdl_pixelformat_bgr24
  let format_rgb888 = i sdl_pixelformat_rgb888
  let format_rgbx8888 = i sdl_pixelformat_rgbx8888
  let format_bgr888 = i sdl_pixelformat_bgr888
  let format_bgrx8888 = i sdl_pixelformat_bgrx8888
  let format_argb8888 = i sdl_pixelformat_argb8888
  let format_rgba8888 = i sdl_pixelformat_rgba8888
  let format_abgr8888 = i sdl_pixelformat_abgr8888
  let format_bgra8888 = i sdl_pixelformat_bgra8888
  let format_argb2101010 = i sdl_pixelformat_argb2101010
  let format_yv12 = i sdl_pixelformat_yv12
  let format_iyuv = i sdl_pixelformat_iyuv
  let format_yuy2 = i sdl_pixelformat_yuy2
  let format_uyvy = i sdl_pixelformat_uyvy
  let format_yvyu = i sdl_pixelformat_yvyu
end

(* Note. Giving direct access to the palette field of SDL_PixelFormat
   is problematic. We can't ensure the pointer won't become invalid at
   a certain point. *)

type _pixel_format
type pixel_format_struct = _pixel_format structure
let pixel_format_struct : pixel_format_struct typ = structure "SDL_PixelFormat"
let pf_format = field pixel_format_struct "format" uint32_t
let pf_palette = field pixel_format_struct "palette" palette
let pf_bits_per_pixel = field pixel_format_struct "BitsPerPixel" uint8_t
let pf_bytes_per_pixel = field pixel_format_struct "BytesPerPixel" uint8_t
let _ = field pixel_format_struct "padding" uint16_t
let _ = field pixel_format_struct "Rmask" uint32_t
let _ = field pixel_format_struct "Gmask" uint32_t
let _ = field pixel_format_struct "Bmask" uint32_t
let _ = field pixel_format_struct "Amask" uint32_t
let _ = field pixel_format_struct "Rloss" uint8_t
let _ = field pixel_format_struct "Gloss" uint8_t
let _ = field pixel_format_struct "Bloss" uint8_t
let _ = field pixel_format_struct "Aloss" uint8_t
let _ = field pixel_format_struct "Rshift" uint8_t
let _ = field pixel_format_struct "Gshift" uint8_t
let _ = field pixel_format_struct "Bshift" uint8_t
let _ = field pixel_format_struct "Ashift" uint8_t
let _ = field pixel_format_struct "refcount" int
let _ = field pixel_format_struct "next" (ptr pixel_format_struct)
let () = seal pixel_format_struct

type pixel_format = pixel_format_struct ptr
let pixel_format : pixel_format typ = ptr pixel_format_struct
let pixel_format_opt : pixel_format option typ = ptr_opt pixel_format_struct

let unsafe_pixel_format_of_ptr addr : pixel_format =
  from_voidp pixel_format_struct (ptr_of_raw_address addr)
let unsafe_ptr_of_pixel_format pixel_format =
  raw_address_of_ptr (to_voidp pixel_format)

let alloc_format =
  foreign "SDL_AllocFormat"
    (uint32_t @-> returning (some_to_ok pixel_format_opt))

let free_format =
  foreign "SDL_FreeFormat" (pixel_format @-> returning void)

let get_pixel_format_name =
  foreign "SDL_GetPixelFormatName" (uint32_t @-> returning string)

let get_pixel_format_format pf =
  getf (!@ pf) pf_format

let get_pixel_format_bits_pp pf =
  Unsigned.UInt8.to_int (getf (!@ pf) pf_bits_per_pixel)

let get_pixel_format_bytes_pp pf =
  Unsigned.UInt8.to_int (getf (!@ pf) pf_bytes_per_pixel)

let get_rgb =
  foreign "SDL_GetRGB"
    (int32_as_uint32_t @-> pixel_format @-> ptr uint8_t @->
     ptr uint8_t @-> ptr uint8_t @-> returning void)

let get_rgb pf p =
  let alloc () = allocate uint8_t Unsigned.UInt8.zero in
  let to_int = Unsigned.UInt8.to_int in
  let r, g, b = alloc (), alloc (), alloc () in
  get_rgb p pf r g b;
   to_int (!@ r), to_int (!@ g), to_int (!@ b)

let get_rgba =
  foreign "SDL_GetRGBA"
    (int32_as_uint32_t @-> pixel_format @-> ptr uint8_t @->
     ptr uint8_t @-> ptr uint8_t @-> ptr uint8_t @-> returning void)

let get_rgba pf p =
  let alloc () = allocate uint8_t Unsigned.UInt8.zero in
  let to_int = Unsigned.UInt8.to_int in
  let r, g, b, a = alloc (), alloc (), alloc (), alloc () in
  get_rgba p pf r g b a;
   to_int (!@ r), to_int (!@ g), to_int (!@ b), to_int (!@ a)

let map_rgb =
  foreign "SDL_MapRGB"
    (pixel_format @-> int_as_uint8_t @-> int_as_uint8_t @-> int_as_uint8_t @->
     returning int32_as_uint32_t)

let map_rgba =
  foreign "SDL_MapRGBA"
    (pixel_format @-> int_as_uint8_t @-> int_as_uint8_t @-> int_as_uint8_t @->
     int_as_uint8_t @-> returning int32_as_uint32_t)

let masks_to_pixel_format_enum =
  foreign "SDL_MasksToPixelFormatEnum"
    (int @-> int32_as_uint32_t @-> int32_as_uint32_t @-> int32_as_uint32_t @->
     int32_as_uint32_t @-> returning uint32_t)

let pixel_format_enum_to_masks =
  foreign "SDL_PixelFormatEnumToMasks"
    (uint32_t @-> ptr int @->
     ptr uint32_t @-> ptr uint32_t @-> ptr uint32_t @-> ptr uint32_t @->
     returning bool)

let pixel_format_enum_to_masks pf =
  let ui () = allocate uint32_t (Unsigned.UInt32.of_int 0) in
  let get iptr = Unsigned.UInt32.to_int32 (!@ iptr) in
  let bpp = allocate int 0 in
  let rm, gm, bm, am = ui (), ui (), ui (), ui () in
  if not (pixel_format_enum_to_masks pf bpp rm gm bm am) then error () else
  Ok (!@ bpp, get rm, get gm, get bm, get am)

let set_pixel_format_palette =
  foreign "SDL_SetPixelFormatPalette"
    (pixel_format @-> palette @-> returning zero_to_ok)

(* Surface *)

type _surface
type surface_struct = _surface structure
let surface_struct : surface_struct typ = structure "SDL_Surface"
let _ = field surface_struct "flags" uint32_t
let surface_format = field surface_struct "format" pixel_format
let surface_w = field surface_struct "w" int
let surface_h = field surface_struct "h" int
let surface_pitch = field surface_struct "pitch" int
let surface_pixels = field surface_struct "pixels" (ptr void)
let _ = field surface_struct "userdata" (ptr void)
let _ = field surface_struct "locked" int
let _ = field surface_struct "lock_data" (ptr void)
let _ = field surface_struct "clip_rect" rect
let _ = field surface_struct "map" (ptr void)
let _ = field surface_struct "refcount" int
let () = seal surface_struct

type surface = surface_struct ptr
let surface : surface typ = ptr surface_struct
let surface_opt : surface option typ = ptr_opt surface_struct

let unsafe_surface_of_ptr addr : surface =
  from_voidp surface_struct (ptr_of_raw_address addr)
let unsafe_ptr_of_surface surface =
  raw_address_of_ptr (to_voidp surface)

let blit_scaled =
  (* SDL_BlitScaled is #ifdef'd to SDL_UpperBlitScaled *)
  foreign "SDL_UpperBlitScaled"
    (surface @-> ptr rect @-> surface @-> ptr rect @-> returning zero_to_ok)

let blit_scaled ~src sr ~dst dr =
  blit_scaled src (addr sr) dst (Rect.opt_addr dr)

let blit_surface =
  (* SDL_BlitSurface is #ifdef'd to SDL_UpperBlit *)
  foreign "SDL_UpperBlit"
    (surface @-> ptr rect @-> surface @-> ptr rect @-> returning zero_to_ok)

let blit_surface ~src sr ~dst dr =
  blit_surface src (Rect.opt_addr sr) dst (Rect.opt_addr dr)

let convert_pixels =
  foreign "SDL_ConvertPixels"
    (int @-> int @-> uint32_t @-> ptr void @-> int @-> uint32_t @->
     ptr void @-> int @-> returning zero_to_ok)

let convert_pixels ~w ~h ~src sp spitch ~dst dp dpitch =
  (* FIXME: we could try check bounds. *)
  let spitch = ba_kind_byte_size (Bigarray.Array1.kind sp) * spitch in
  let dpitch = ba_kind_byte_size (Bigarray.Array1.kind dp) * dpitch in
  let sp = to_voidp (bigarray_start array1 sp) in
  let dp = to_voidp (bigarray_start array1 dp) in
  convert_pixels w h src sp spitch dst dp dpitch

let convert_surface =
  foreign "SDL_ConvertSurface"
    (surface @-> pixel_format @-> uint32_t @->
     returning (some_to_ok surface_opt))

let convert_surface s pf =
  convert_surface s pf Unsigned.UInt32.zero

let convert_surface_format =
  foreign "SDL_ConvertSurfaceFormat"
    (surface @-> uint32_t @-> uint32_t @-> returning (some_to_ok surface_opt))

let convert_surface_format s pf =
  convert_surface_format s pf Unsigned.UInt32.zero

let create_rgb_surface =
  foreign "SDL_CreateRGBSurface"
    (uint32_t @-> int @-> int @-> int @-> int32_as_uint32_t @->
     int32_as_uint32_t @-> int32_as_uint32_t @-> int32_as_uint32_t @->
     returning (some_to_ok surface_opt))

let create_rgb_surface ~w ~h ~depth rmask gmask bmask amask =
  create_rgb_surface Unsigned.UInt32.zero w h depth rmask gmask bmask amask

let create_rgb_surface_from =
  foreign "SDL_CreateRGBSurfaceFrom"
    (ptr void @-> int @-> int @-> int @-> int @-> int32_as_uint32_t @->
     int32_as_uint32_t @-> int32_as_uint32_t @-> int32_as_uint32_t @->
     returning (some_to_ok surface_opt))

let create_rgb_surface_from p ~w ~h ~depth ~pitch rmask gmask bmask amask =
  (* FIXME: we could try check bounds. *)
  let pitch = ba_kind_byte_size (Bigarray.Array1.kind p) * pitch in
  let p = to_voidp (bigarray_start array1 p) in
  create_rgb_surface_from p w h depth pitch rmask gmask bmask amask

let fill_rect =
  foreign "SDL_FillRect"
    (surface @-> ptr rect @-> int32_as_uint32_t @-> returning zero_to_ok)

let fill_rect s r c =
  fill_rect s (Rect.opt_addr r) c

let fill_rects =
  foreign "SDL_FillRects"
    (surface @-> ptr void @-> int @-> int32_as_uint32_t @->
     returning zero_to_ok)

let fill_rects_ba s rs col =
  let len = Bigarray.Array1.dim rs in
  if len mod 4 <> 0 then invalid_arg (err_length_mul len 4) else
  let count = len / 4 in
  let rs = to_voidp (bigarray_start array1 rs) in
  fill_rects s rs count col

let fill_rects s rs col =
  let count = List.length rs in
  let a = CArray.of_list rect rs in
  fill_rects s (to_voidp (CArray.start a)) count col

let free_surface =
  foreign "SDL_FreeSurface" (surface @-> returning void)

let get_clip_rect =
  foreign "SDL_GetClipRect" (surface @-> ptr rect @-> returning void)

let get_clip_rect s =
  let r = make rect in
  (get_clip_rect s (addr r); r)

let get_color_key =
  foreign "SDL_GetColorKey"
    (surface @-> ptr uint32_t @-> returning zero_to_ok)

let get_color_key s =
  let key = allocate uint32_t Unsigned.UInt32.zero in
  match get_color_key s key with
  | Ok () -> Ok (Unsigned.UInt32.to_int32 (!@ key)) | Error _ as e -> e

let get_surface_alpha_mod =
  foreign "SDL_GetSurfaceAlphaMod"
    (surface @-> ptr uint8_t @-> returning zero_to_ok)

let get_surface_alpha_mod s =
  let alpha = allocate uint8_t Unsigned.UInt8.zero in
  match get_surface_alpha_mod s alpha with
  | Ok () -> Ok (Unsigned.UInt8.to_int (!@ alpha)) | Error _ as e -> e

let get_surface_blend_mode =
  foreign "SDL_GetSurfaceBlendMode"
    (surface @-> ptr int @-> returning zero_to_ok)

let get_surface_blend_mode s =
  let mode = allocate int 0 in
  match get_surface_blend_mode s mode with
  Ok () -> Ok (!@ mode) | Error _ as e -> e

let get_surface_color_mod =
  foreign "SDL_GetSurfaceColorMod"
    (surface @-> ptr uint8_t @-> ptr uint8_t @-> ptr uint8_t @->
     returning zero_to_ok)

let get_surface_color_mod s =
  let alloc () = allocate uint8_t Unsigned.UInt8.zero in
  let get v = Unsigned.UInt8.to_int (!@ v) in
  let r, g, b = alloc (), alloc (), alloc () in
  match get_surface_color_mod s r g b with
  | Ok () -> Ok (get r, get g, get b) | Error _ as e -> e

let get_surface_format_enum s =
  (* We don't give direct access to the format field. This prevents
     memory ownership problems. *)
  get_pixel_format_format (getf (!@ s) surface_format)

let get_surface_pitch s =
  getf (!@ s) surface_pitch

let get_surface_pixels s kind =
  let pitch = get_surface_pitch s in
  let kind_size = ba_kind_byte_size kind in
  if pitch mod kind_size <> 0
  then invalid_arg (err_bigarray_pitch pitch kind_size)
  else
  let h = getf (!@ s) surface_h in
  let ba_size = (pitch * h) / kind_size in
  let pixels = getf (!@ s) surface_pixels in
  let pixels = coerce (ptr void) (access_ptr_typ_of_ba_kind kind) pixels in
  bigarray_of_ptr array1 ba_size kind pixels

let get_surface_size s =
  getf (!@ s) surface_w, getf (!@ s) surface_h

let load_bmp_rw =
  foreign "SDL_LoadBMP_RW"
    (rw_ops @-> bool @-> returning (some_to_ok surface_opt))

let load_bmp_rw rw ~close =
  load_bmp_rw rw close

let load_bmp file =
  (* SDL_LoadBMP is cpp based *)
  match rw_from_file file "rb" with
  | Error _ as e -> e
  | Ok rw -> load_bmp_rw rw ~close:true

let lock_surface =
  foreign "SDL_LockSurface" (surface @-> returning zero_to_ok)

let lower_blit =
  foreign "SDL_LowerBlit"
    (surface @-> ptr rect @-> surface @-> ptr rect @-> returning zero_to_ok)

let lower_blit ~src sr ~dst dr =
  lower_blit src (addr sr) dst (addr dr)

let lower_blit_scaled =
  foreign "SDL_LowerBlitScaled"
    (surface @-> ptr rect @-> surface @-> ptr rect @-> returning zero_to_ok)

let lower_blit_scaled ~src sr ~dst dr =
  lower_blit_scaled src (addr sr) dst (addr dr)

let save_bmp_rw =
  foreign "SDL_SaveBMP_RW"
    (surface @-> rw_ops @-> bool @-> returning zero_to_ok)

let save_bmp_rw s rw ~close =
  save_bmp_rw s rw close

let save_bmp s file =
  (* SDL_SaveBMP is cpp based *)
  match rw_from_file file "wb" with
  | Error _ as e -> e
  | Ok rw -> save_bmp_rw s rw ~close:true

let set_clip_rect =
  foreign "SDL_SetClipRect" (surface @-> ptr rect @-> returning bool)

let set_clip_rect s r =
  set_clip_rect s (addr r)

let set_color_key =
  foreign "SDL_SetColorKey"
    (surface @-> bool @-> int32_as_uint32_t @-> returning zero_to_ok)

let set_surface_alpha_mod =
  foreign "SDL_SetSurfaceAlphaMod"
    (surface @-> int_as_uint8_t @-> returning zero_to_ok)

let set_surface_blend_mode =
  foreign "SDL_SetSurfaceBlendMode"
    (surface @-> int @-> returning zero_to_ok)

let set_surface_color_mod =
  foreign "SDL_SetSurfaceColorMod"
    (surface @-> int_as_uint8_t @-> int_as_uint8_t @-> int_as_uint8_t @->
     returning zero_to_ok)

let set_surface_palette =
  foreign "SDL_SetSurfacePalette"
    (surface @-> palette @-> returning zero_to_ok)

let set_surface_rle =
  foreign "SDL_SetSurfaceRLE" (surface @-> bool @-> returning zero_to_ok)

let unlock_surface =
  foreign "SDL_UnlockSurface" (surface @-> returning void)

(* Renderers *)

type flip = int
module Flip = struct
  let ( + ) = ( lor )
  let none = sdl_flip_none
  let horizontal = sdl_flip_horizontal
  let vertical = sdl_flip_vertical
end

type texture = unit ptr
let texture : texture typ = ptr void
let texture_opt : texture option typ = ptr_opt void

let unsafe_texture_of_ptr addr : texture =
  ptr_of_raw_address addr
let unsafe_ptr_of_texture texture =
  raw_address_of_ptr (to_voidp texture)

type renderer = unit ptr
let renderer : renderer typ = ptr void
let renderer_opt : renderer option typ = ptr_opt void

let unsafe_renderer_of_ptr addr : renderer =
  ptr_of_raw_address addr
let unsafe_ptr_of_renderer renderer =
  raw_address_of_ptr (to_voidp renderer)

module Renderer = struct
  type flags = Unsigned.uint32
  let i = Unsigned.UInt32.of_int
  let ( + ) = Unsigned.UInt32.logor
  let test f m = Unsigned.UInt32.(compare (logand f m) zero <> 0)
  let eq f f' = Unsigned.UInt32.(compare f f' = 0)
  let none = Unsigned.UInt32.zero
  let software = i sdl_renderer_software
  let accelerated = i sdl_renderer_accelerated
  let presentvsync = i sdl_renderer_presentvsync
  let targettexture = i sdl_renderer_targettexture
end

type renderer_info =
  { ri_name : string;
    ri_flags : Renderer.flags;
    ri_texture_formats : Pixel.format_enum list;
    ri_max_texture_width : int;
    ri_max_texture_height : int; }

let renderer_info = structure "SDL_RendererInfo"
let ri_name = field renderer_info "name" string
let ri_flags = field renderer_info "flags" uint32_t
let ri_num_tf = field renderer_info "num_texture_formats" uint32_t
let ri_tfs = field renderer_info "texture_formats" (array 16 uint32_t)
let ri_max_texture_width = field renderer_info "max_texture_width" int
let ri_max_texture_height = field renderer_info "max_texture_height" int
let () = seal renderer_info

let renderer_info_of_c c =
  let ri_name = getf c ri_name in
  let ri_flags = getf c ri_flags in
  let num_tf = Unsigned.UInt32.to_int (getf c ri_num_tf) in
  let tfs = getf c ri_tfs in
  let ri_texture_formats =
    let acc = ref [] in
    for i = 0 to num_tf - 1 do acc := (CArray.get tfs i) :: !acc done;
    List.rev !acc
  in
  let ri_max_texture_width = getf c ri_max_texture_width in
  let ri_max_texture_height = getf c ri_max_texture_height in
  { ri_name; ri_flags; ri_texture_formats; ri_max_texture_width;
    ri_max_texture_height }

let create_renderer =
  foreign "SDL_CreateRenderer"
    (window @-> int @-> uint32_t @-> returning (some_to_ok renderer_opt))

let create_renderer ?(index = -1) ?(flags = Unsigned.UInt32.zero) w =
  create_renderer w index flags

let create_software_renderer =
  foreign "SDL_CreateSoftwareRenderer"
    (surface @-> returning (some_to_ok renderer_opt))

let destroy_renderer =
  foreign "SDL_DestroyRenderer" (renderer @-> returning void)

let get_num_render_drivers =
  foreign "SDL_GetNumRenderDrivers" (void @-> returning nat_to_ok)

let get_render_draw_blend_mode =
  foreign "SDL_GetRenderDrawBlendMode"
    (renderer @-> ptr int @-> returning zero_to_ok)

let get_render_draw_blend_mode r =
  let m = allocate int 0 in
  match get_render_draw_blend_mode r m with
  | Ok () -> Ok !@m | Error _ as e -> e

let get_render_draw_color =
  foreign "SDL_GetRenderDrawColor"
    (renderer @-> ptr uint8_t @-> ptr uint8_t @-> ptr uint8_t @->
     ptr uint8_t @-> returning zero_to_ok)

let get_render_draw_color rend =
  let alloc () = allocate uint8_t Unsigned.UInt8.zero in
  let get v = Unsigned.UInt8.to_int (!@ v) in
  let r, g, b, a = alloc (), alloc (), alloc (), alloc () in
  match get_render_draw_color rend r g b a with
  | Ok () -> Ok (get r, get g, get b, get a) | Error _ as e -> e

let get_render_driver_info =
  foreign "SDL_GetRenderDriverInfo"
    (int @-> ptr renderer_info @-> returning zero_to_ok)

let get_render_driver_info i =
  let info = make renderer_info in
  match get_render_driver_info i (addr info) with
  | Ok () -> Ok (renderer_info_of_c info) | Error _ as e -> e

let get_render_target =
  foreign "SDL_GetRenderTarget" (renderer @-> returning texture_opt)

let get_renderer =
  foreign "SDL_GetRenderer"
    (window @-> returning (some_to_ok renderer_opt))

let get_renderer_info =
  foreign "SDL_GetRendererInfo"
    (renderer @-> ptr renderer_info @-> returning zero_to_ok)

let get_renderer_info r =
  let info = make renderer_info in
  match get_renderer_info r (addr info) with
  | Ok () -> Ok (renderer_info_of_c info) | Error _ as e -> e

let get_renderer_output_size =
  foreign "SDL_GetRendererOutputSize"
    (renderer @-> ptr int @-> ptr int @-> returning zero_to_ok)

let get_renderer_output_size r =
  let w = allocate int 0 in
  let h = allocate int 0 in
  match get_renderer_output_size r w h with
  | Ok () -> Ok (!@ w, !@ h) | Error _ as e -> e

let render_clear =
  foreign "SDL_RenderClear" (renderer @-> returning zero_to_ok)

let render_copy =
  foreign "SDL_RenderCopy"
    (renderer @-> texture @-> ptr rect @-> ptr rect @->
     returning zero_to_ok)

let render_copy ?src ?dst r t =
  render_copy r t (Rect.opt_addr src) (Rect.opt_addr dst)

let render_copy_ex =
  foreign "SDL_RenderCopyEx"
    (renderer @-> texture @-> ptr rect @-> ptr rect @-> double @->
     ptr point @-> int @-> returning zero_to_ok)

let render_copy_ex ?src ?dst r t angle c flip =
  render_copy_ex r t (Rect.opt_addr src) (Rect.opt_addr dst) angle
    (Point.opt_addr c) flip

let render_draw_line =
  foreign "SDL_RenderDrawLine"
    (renderer @-> int @-> int @-> int @-> int @-> returning zero_to_ok)

let render_draw_lines =
  foreign "SDL_RenderDrawLines"
    (renderer @-> ptr void @-> int @-> returning zero_to_ok)

let render_draw_lines_ba r ps =
  let len = Bigarray.Array1.dim ps in
  if len mod 2 <> 0 then invalid_arg (err_length_mul len 2) else
  let count = len / 2 in
  let ps = to_voidp (bigarray_start array1 ps) in
  render_draw_lines r ps count

let render_draw_lines r ps =
  let count = List.length ps in
  let a = CArray.of_list point ps in
  render_draw_lines r (to_voidp (CArray.start a)) count

let render_draw_point =
  foreign "SDL_RenderDrawPoint"
    (renderer @-> int @-> int @-> returning zero_to_ok)

let render_draw_points =
  foreign "SDL_RenderDrawPoints"
    (renderer @-> ptr void @-> int @-> returning zero_to_ok)

let render_draw_points_ba r ps =
  let len = Bigarray.Array1.dim ps in
  if len mod 2 <> 0 then invalid_arg (err_length_mul len 2) else
  let count = len / 2 in
  let ps = to_voidp (bigarray_start array1 ps) in
  render_draw_points r ps count

let render_draw_points r ps =
  let count = List.length ps in
  let a = CArray.of_list point ps in
  render_draw_points r (to_voidp (CArray.start a)) count

let render_draw_rect =
  foreign "SDL_RenderDrawRect"
    (renderer @-> ptr rect @-> returning zero_to_ok)

let render_draw_rect rend r =
  render_draw_rect rend (Rect.opt_addr r)

let render_draw_rects =
  foreign "SDL_RenderDrawRects"
    (renderer @-> ptr void @-> int @-> returning zero_to_ok)

let render_draw_rects_ba r rs =
  let len = Bigarray.Array1.dim rs in
  if len mod 4 <> 0 then invalid_arg (err_length_mul len 4) else
  let count = len / 4 in
  let rs = to_voidp (bigarray_start array1 rs) in
  render_draw_rects r rs count

let render_draw_rects r rs =
  let count = List.length rs in
  let a = CArray.of_list rect rs in
  render_draw_rects r (to_voidp (CArray.start a)) count

let render_fill_rect =
  foreign "SDL_RenderFillRect"
    (renderer @-> ptr rect @-> returning zero_to_ok)

let render_fill_rect rend r =
  render_fill_rect rend (Rect.opt_addr r)

let render_fill_rects =
  foreign "SDL_RenderFillRects"
    (renderer @-> ptr void @-> int @-> returning zero_to_ok)

let render_fill_rects_ba r rs =
  let len = Bigarray.Array1.dim rs in
  if len mod 4 <> 0 then invalid_arg (err_length_mul len 4) else
  let count = len / 4 in
  let rs = to_voidp (bigarray_start array1 rs) in
  render_fill_rects r rs count

let render_fill_rects r rs =
  let count = List.length rs in
  let a = CArray.of_list rect rs in
  render_fill_rects r (to_voidp (CArray.start a)) count

let render_get_clip_rect =
  foreign "SDL_RenderGetClipRect"
    (renderer @-> ptr rect @-> returning void)

let render_get_clip_rect rend =
  let r = make rect in
  render_get_clip_rect rend (addr r);
  r

let render_get_logical_size =
  foreign "SDL_RenderGetLogicalSize"
    (renderer @-> ptr int @-> ptr int @-> returning void)

let render_get_logical_size r =
  let w = allocate int 0 in
  let h = allocate int 0 in
  render_get_logical_size r w h;
  !@ w, !@ h

let render_get_scale =
  foreign "SDL_RenderGetScale"
    (renderer @-> ptr float @-> ptr float @-> returning void)

let render_get_scale r =
  let x = allocate float 0. in
  let y = allocate float 0. in
  render_get_scale r x y;
  !@ x, !@ y

let render_get_viewport =
  foreign "SDL_RenderGetViewport"
    (renderer @-> ptr rect @-> returning void)

let render_get_viewport rend =
  let r = make rect in
  render_get_viewport rend (addr r);
  r

let render_present =
  foreign "SDL_RenderPresent" (renderer @-> returning void)

let render_read_pixels =
  foreign "SDL_RenderReadPixels"
    (renderer @-> ptr rect @-> uint32_t @-> ptr void @-> int @->
     returning zero_to_ok)

let render_read_pixels r rect format pixels pitch =
  let format = match format with None -> Unsigned.UInt32.zero | Some f -> f in
  let pixels = to_voidp (bigarray_start array1 pixels) in
  render_read_pixels r (Rect.opt_addr rect) format pixels pitch

let render_set_clip_rect =
  foreign "SDL_RenderSetClipRect"
    (renderer @-> ptr rect @-> returning zero_to_ok)

let render_set_clip_rect rend r =
  render_set_clip_rect rend (Rect.opt_addr r)

let render_set_logical_size =
  foreign "SDL_RenderSetLogicalSize"
    (renderer @-> int @-> int @-> returning zero_to_ok)

let render_set_scale =
  foreign "SDL_RenderSetScale"
    (renderer @-> float @-> float @-> returning zero_to_ok)

let render_set_viewport =
  foreign "SDL_RenderSetViewport"
    (renderer @-> ptr rect @-> returning zero_to_ok)

let render_set_viewport rend r =
  render_set_viewport rend (Rect.opt_addr r)

let render_target_supported =
  foreign "SDL_RenderTargetSupported" (renderer @-> returning bool)

let set_render_draw_blend_mode =
  foreign "SDL_SetRenderDrawBlendMode"
    (renderer @-> int @-> returning zero_to_ok)

let set_render_draw_color =
  foreign "SDL_SetRenderDrawColor"
    (renderer @-> int_as_uint8_t @-> int_as_uint8_t @-> int_as_uint8_t @->
     int_as_uint8_t @-> returning zero_to_ok)

let set_render_target =
  foreign "SDL_SetRenderTarget"
    (renderer @-> texture @-> returning zero_to_ok)

let set_render_target r t =
  let t = match t with None -> null | Some t -> t in
  set_render_target r t

(* Textures *)

module Texture = struct
  type access = int
  let access_static = sdl_textureaccess_static
  let access_streaming = sdl_textureaccess_streaming
  let access_target = sdl_textureaccess_target

  let i = Unsigned.UInt32.of_int
  type modulate = Unsigned.uint32
  let modulate_none = i sdl_texturemodulate_none
  let modulate_color = i sdl_texturemodulate_color
  let modulate_alpha = i sdl_texturemodulate_alpha
end

let create_texture =
  foreign "SDL_CreateTexture"
    (renderer @-> uint32_t @-> int @-> int @-> int @->
     returning (some_to_ok renderer_opt))

let create_texture r pf access ~w ~h =
  create_texture r pf access w h

let create_texture_from_surface =
  foreign "SDL_CreateTextureFromSurface"
    (renderer @-> surface @-> returning (some_to_ok texture_opt))

let destroy_texture =
  foreign "SDL_DestroyTexture" (texture @-> returning void)

let get_texture_alpha_mod =
  foreign "SDL_GetTextureAlphaMod"
    (texture @-> ptr uint8_t @-> returning zero_to_ok)

let get_texture_alpha_mod t =
  let alpha = allocate uint8_t Unsigned.UInt8.zero in
  match get_texture_alpha_mod t alpha with
  | Ok () -> Ok (Unsigned.UInt8.to_int (!@ alpha)) | Error _ as e -> e

let get_texture_blend_mode =
  foreign "SDL_GetTextureBlendMode"
    (texture @-> ptr int @-> returning zero_to_ok)

let get_texture_blend_mode t =
  let m = allocate int 0 in
  match get_texture_blend_mode t m with
  | Ok () -> Ok (!@ m) | Error _ as e -> e

let get_texture_color_mod =
  foreign "SDL_GetTextureColorMod"
    (renderer @-> ptr uint8_t @-> ptr uint8_t @-> ptr uint8_t @->
     returning zero_to_ok)

let get_texture_color_mod t =
  let alloc () = allocate uint8_t Unsigned.UInt8.zero in
  let get v = Unsigned.UInt8.to_int (!@ v) in
  let r, g, b = alloc (), alloc (), alloc () in
  match get_texture_color_mod t r g b with
  | Ok () -> Ok (get r, get g, get b) | Error _ as e -> e

let query_texture =
  foreign "SDL_QueryTexture"
    (texture @-> ptr uint32_t @-> ptr int @-> ptr int @-> ptr int @->
     returning zero_to_ok)

let _texture_height t =
  let h = allocate int 0 in
  let unull = coerce (ptr void) (ptr uint32_t) null in
  let inull = coerce (ptr void) (ptr int) null in
  match query_texture t unull inull inull h with
  | Ok () -> Ok (!@ h) | Error _ as e -> e

let lock_texture =
  foreign "SDL_LockTexture"
    (texture @-> ptr rect @-> ptr (ptr void) @-> ptr int @->
     returning zero_to_ok)

let lock_texture t r kind =
  match (match r with None -> _texture_height t | Some r -> Ok (Rect.h r)) with
  | Error _ as e -> e
  | Ok h ->
      let pitch = allocate int 0 in
      let p = allocate (ptr void) null in
      match lock_texture t (Rect.opt_addr r) p pitch with
      | Error _ as e -> e
      | Ok () ->
          let p = !@ p in
          let pitch = !@ pitch in
          let kind_size = ba_kind_byte_size kind in
          if pitch mod kind_size <> 0
          then invalid_arg (err_bigarray_pitch pitch kind_size)
          else
          let ba_size = (pitch * h) / kind_size in
          let pixels = coerce (ptr void) (access_ptr_typ_of_ba_kind kind) p in
          Ok (bigarray_of_ptr array1 ba_size kind pixels, pitch / kind_size)

let query_texture t =
  let pf = allocate uint32_t Unsigned.UInt32.zero in
  let access = allocate int 0 in
  let w = allocate int 0 in
  let h = allocate int 0 in
  match query_texture t pf access w h with
  | Ok () -> Ok (!@ pf, !@ access, (!@ w, !@ h)) | Error _ as e -> e

let set_texture_alpha_mod =
  foreign "SDL_SetTextureAlphaMod"
    (texture @-> int_as_uint8_t @-> returning zero_to_ok)

let set_texture_blend_mode =
  foreign "SDL_SetTextureBlendMode"
    (texture @-> int @-> returning zero_to_ok)

let set_texture_color_mod =
  foreign "SDL_SetTextureColorMod"
    (texture @-> int_as_uint8_t @-> int_as_uint8_t @-> int_as_uint8_t @->
     returning zero_to_ok)

let unlock_texture =
  foreign "SDL_UnlockTexture" (texture @-> returning void)

let update_texture =
  foreign "SDL_UpdateTexture"
    (texture @-> ptr rect @-> ptr void @-> int @-> returning zero_to_ok)

let update_texture t rect pixels pitch =
  let pitch = pitch * (ba_kind_byte_size (Bigarray.Array1.kind pixels)) in
  let pixels = to_voidp (bigarray_start array1 pixels) in
  update_texture t (Rect.opt_addr rect) pixels pitch

let update_yuv_texture =
  foreign "SDL_UpdateYUVTexture"
    (texture @-> ptr rect @->
     ptr void @-> int @-> ptr void @-> int @-> ptr void @-> int @->
     returning zero_to_ok)

let update_yuv_texture r rect ~y ypitch ~u upitch ~v vpitch =
  let yp = to_voidp (bigarray_start array1 y) in
  let up = to_voidp (bigarray_start array1 u) in
  let vp = to_voidp (bigarray_start array1 v) in
  update_yuv_texture r (Rect.opt_addr rect) yp ypitch up upitch vp vpitch

(* Video drivers *)

let get_current_video_driver =
  foreign "SDL_GetCurrentVideoDriver" (void @-> returning string_opt)

let get_num_video_drivers =
  foreign "SDL_GetNumVideoDrivers" (void @-> returning nat_to_ok)

let get_video_driver =
  foreign "SDL_GetVideoDriver" (int @-> returning (some_to_ok string_opt))

let video_init =
  foreign "SDL_VideoInit" (string_opt @-> returning zero_to_ok)

let video_quit =
  foreign "SDL_VideoQuit" (void @-> returning void)

(* Displays *)

type driverdata = unit ptr
let driverdata = ptr_opt void

type display_mode =
  { dm_format : Pixel.format_enum;
    dm_w : int;
    dm_h : int;
    dm_refresh_rate : int option;
    dm_driverdata : driverdata option }

type _display_mode
let display_mode : _display_mode structure typ = structure "SDL_DisplayMode"
let dm_format = field display_mode "format" uint32_t
let dm_w = field display_mode "w" int
let dm_h = field display_mode "h" int
let dm_refresh_rate = field display_mode "refresh_rate" int
let dm_driverdata = field display_mode "driverdata" driverdata
let () = seal display_mode

let display_mode_to_c o =
  let c = make display_mode in
  let rate = match o.dm_refresh_rate with None -> 0 | Some r -> r in
  setf c dm_format o.dm_format;
  setf c dm_w o.dm_w;
  setf c dm_h o.dm_h;
  setf c dm_refresh_rate rate;
  setf c dm_driverdata o.dm_driverdata;
  c

let display_mode_of_c c =
  let dm_format = getf c dm_format in
  let dm_w = getf c dm_w in
  let dm_h = getf c dm_h in
  let dm_refresh_rate = match getf c dm_refresh_rate with
  | 0 -> None | r -> Some r
  in
  let dm_driverdata = getf c dm_driverdata in
  { dm_format; dm_w; dm_h; dm_refresh_rate; dm_driverdata }

let get_closest_display_mode =
  foreign "SDL_GetClosestDisplayMode"
    (int @-> ptr display_mode @-> ptr display_mode @->
       returning (ptr_opt void))

let get_closest_display_mode i m =
  let mode = display_mode_to_c m in
  let closest = make display_mode in
  match get_closest_display_mode i (addr mode) (addr closest) with
  | None -> None
  | Some _ -> Some (display_mode_of_c closest)

let get_current_display_mode =
  foreign "SDL_GetCurrentDisplayMode"
    (int @-> ptr display_mode @-> returning zero_to_ok)

let get_current_display_mode i =
  let mode = make display_mode in
  match get_current_display_mode i (addr mode) with
  | Ok () -> Ok (display_mode_of_c mode) | Error _ as e -> e

let get_desktop_display_mode =
  foreign "SDL_GetDesktopDisplayMode"
    (int @-> ptr display_mode @-> returning zero_to_ok)

let get_desktop_display_mode i =
  let mode = make display_mode in
  match get_desktop_display_mode i (addr mode) with
  | Ok () -> Ok (display_mode_of_c mode) | Error _ as e -> e

let get_display_bounds =
  foreign "SDL_GetDisplayBounds"
    (int @-> ptr rect @-> returning zero_to_ok)

let get_display_bounds i =
  let r = make rect in
  match get_display_bounds i (addr r) with
  | Ok () -> Ok r | Error _ as e -> e

let get_display_mode =
  foreign "SDL_GetDisplayMode"
    (int @-> int @-> ptr display_mode @-> returning zero_to_ok)

let get_display_mode d i =
  let mode = make display_mode in
  match get_display_mode d i (addr mode) with
  | Ok () -> Ok (display_mode_of_c mode) | Error _ as e -> e

let get_num_display_modes =
  foreign "SDL_GetNumDisplayModes" (int @-> returning nat_to_ok)

let get_display_name =
  foreign "SDL_GetDisplayName" (int @-> returning (some_to_ok string_opt))

let get_num_video_displays =
  foreign "SDL_GetNumVideoDisplays" (void @-> returning nat_to_ok)

(* Windows *)

module Window = struct
  let pos_undefined = sdl_windowpos_undefined
  let pos_centered = sdl_windowpos_centered

  type flags = Unsigned.uint32
  let i = Unsigned.UInt32.of_int
  let ( + ) = Unsigned.UInt32.logor
  let test f m = Unsigned.UInt32.(compare (logand f m) zero <> 0)
  let eq f f' = Unsigned.UInt32.(compare f f' = 0)
  let windowed = i 0
  let fullscreen = i sdl_window_fullscreen
  let fullscreen_desktop = i sdl_window_fullscreen_desktop
  let opengl = i sdl_window_opengl
  let shown = i sdl_window_shown
  let shown = i sdl_window_shown
  let hidden = i sdl_window_hidden
  let borderless = i sdl_window_borderless
  let resizable = i sdl_window_resizable
  let minimized = i sdl_window_minimized
  let maximized = i sdl_window_maximized
  let input_grabbed = i sdl_window_input_grabbed
  let input_focus = i sdl_window_input_focus
  let mouse_focus = i sdl_window_mouse_focus
  let foreign = i sdl_window_foreign
  let allow_highdpi = i sdl_window_allow_highdpi
end

let create_window =
  foreign "SDL_CreateWindow"
    (string @-> int @-> int @-> int @-> int @-> uint32_t @->
     returning (some_to_ok window_opt))

let create_window t ?(x = Window.pos_undefined) ?(y = Window.pos_undefined)
    ~w ~h flags = create_window t x y w h flags

let create_window_and_renderer =
  foreign "SDL_CreateWindowAndRenderer"
    (int @-> int @-> uint32_t @-> ptr window @-> ptr renderer @->
     (returning zero_to_ok))

let create_window_and_renderer ~w ~h flags =
  let win = allocate window null in
  let r = allocate renderer null in
  match create_window_and_renderer w h flags win r with
  | Ok () -> Ok (!@ win, !@ r) | Error _ as e -> e

let destroy_window =
  foreign "SDL_DestroyWindow" (window @-> returning void)

let get_window_brightness =
  foreign "SDL_GetWindowBrightness" (window @-> returning float)

let get_window_display_index =
  foreign "SDL_GetWindowDisplayIndex" (window @-> returning nat_to_ok)

let get_window_display_mode =
  foreign "SDL_GetWindowDisplayMode"
    (window @-> (ptr display_mode) @-> returning int)

let get_window_display_mode w =
  let mode = make display_mode in
  match get_window_display_mode w (addr mode) with
  | 0 -> Ok (display_mode_of_c mode) | err -> error ()

let get_window_flags =
  foreign "SDL_GetWindowFlags" (window @-> returning uint32_t)

let get_window_from_id =
  foreign "SDL_GetWindowFromID"
    (int_as_uint32_t @-> returning (some_to_ok window_opt))

let get_window_gamma_ramp =
  foreign "SDL_GetWindowGammaRamp"
    (window @-> ptr void @-> ptr void @-> ptr void @-> returning zero_to_ok)

let get_window_gamma_ramp w =
  let create_ramp () = ba_create Bigarray.int16_unsigned 256 in
  let r, g, b = create_ramp (), create_ramp (), create_ramp () in
  let ramp_ptr r = to_voidp (bigarray_start array1 r) in
  match get_window_gamma_ramp w (ramp_ptr r) (ramp_ptr g) (ramp_ptr b) with
  | Ok () -> Ok (r, g, b) | Error _ as e -> e

let get_window_grab =
  foreign "SDL_GetWindowGrab" (window @-> returning bool)

let get_window_id =
  foreign "SDL_GetWindowID" (window @-> returning int_as_uint32_t)

let get_window_maximum_size =
  foreign "SDL_GetWindowMaximumSize"
    (window @-> (ptr int) @-> (ptr int) @-> returning void)

let get_window_maximum_size win =
  let w = allocate int 0 in
  let h = allocate int 0 in
  get_window_maximum_size win w h;
  !@ w, !@ h

let get_window_minimum_size =
  foreign "SDL_GetWindowMinimumSize"
    (window @-> (ptr int) @-> (ptr int) @-> returning void)

let get_window_minimum_size win =
  let w = allocate int 0 in
  let h = allocate int 0 in
  get_window_minimum_size win w h;
  !@ w, !@ h

let get_window_pixel_format =
  foreign "SDL_GetWindowPixelFormat" (window @-> returning uint32_t)

let get_window_position =
  foreign "SDL_GetWindowPosition"
    (window @-> (ptr int) @-> (ptr int) @-> returning void)

let get_window_position win =
  let x = allocate int 0 in
  let y = allocate int 0 in
  get_window_position win x y;
  !@ x, !@ y

let get_window_size =
  foreign "SDL_GetWindowSize"
    (window @-> (ptr int) @-> (ptr int) @-> returning void)

let get_window_size win =
  let w = allocate int 0 in
  let h = allocate int 0 in
  get_window_size win w h;
  !@ w, !@ h

let get_window_surface =
  foreign "SDL_GetWindowSurface"
    (window @-> returning (some_to_ok surface_opt))

let get_window_title =
  foreign "SDL_GetWindowTitle" (window @-> returning string)

let hide_window =
  foreign "SDL_HideWindow" (window @-> returning void)

let maximize_window =
  foreign "SDL_MaximizeWindow" (window @-> returning void)

let minimize_window =
  foreign "SDL_MinimizeWindow" (window @-> returning void)

let raise_window =
  foreign "SDL_RaiseWindow" (window @-> returning void)

let restore_window =
  foreign "SDL_RestoreWindow" (window @-> returning void)

let set_window_bordered =
  foreign "SDL_SetWindowBordered" (window @-> bool @-> returning void)

let set_window_brightness =
  foreign "SDL_SetWindowBrightness"
    (window @-> float @-> returning zero_to_ok)

let set_window_display_mode =
  foreign "SDL_SetWindowDisplayMode"
    (window @-> (ptr display_mode) @-> returning zero_to_ok)

let set_window_display_mode w m =
  let mode = display_mode_to_c m in
  set_window_display_mode w (addr mode)

let set_window_fullscreen =
  foreign "SDL_SetWindowFullscreen"
    (window @-> uint32_t @-> returning zero_to_ok)

let set_window_gamma_ramp =
  foreign "SDL_SetWindowGammaRamp"
    (window @-> ptr void @-> ptr void @-> ptr void @->
     returning zero_to_ok)

let set_window_gamma_ramp w r g b =
  let ramp_ptr r = to_voidp (bigarray_start array1 r) in
  set_window_gamma_ramp w (ramp_ptr r) (ramp_ptr g) (ramp_ptr b)

let set_window_grab =
  foreign "SDL_SetWindowGrab" (window @-> bool @-> returning void)

let set_window_icon =
  foreign "SDL_SetWindowIcon" (window @-> surface @-> returning void)

let set_window_maximum_size =
  foreign "SDL_SetWindowMaximumSize"
    (window @-> int @-> int @-> returning void)

let set_window_maximum_size win ~w ~h =
  set_window_maximum_size win w h

let set_window_minimum_size =
  foreign "SDL_SetWindowMinimumSize"
    (window @-> int @-> int @-> returning void)

let set_window_minimum_size win ~w ~h =
  set_window_minimum_size win w h

let set_window_position =
  foreign "SDL_SetWindowPosition"
    (window @-> int @-> int @-> returning void)

let set_window_position win ~x ~y =
  set_window_position win x y

let set_window_size =
  foreign "SDL_SetWindowSize" (window @-> int @-> int @-> returning void)

let set_window_size win ~w ~h =
  set_window_size win w h

let set_window_title =
  foreign "SDL_SetWindowTitle" (window @-> string @-> returning void)

let show_window =
  foreign "SDL_ShowWindow" (window @-> returning void)

let update_window_surface =
  foreign "SDL_UpdateWindowSurface" (window @-> returning zero_to_ok)

let update_window_surface_rects =
  foreign "SDL_UpdateWindowSurfaceRects"
    (window @-> ptr void @-> int @-> returning zero_to_ok)

let update_window_surface_rects_ba w rs =
  let len = Bigarray.Array1.dim rs in
  if len mod 4 <> 0 then invalid_arg (err_length_mul len 4) else
  let count = len / 4 in
  let rs = to_voidp (bigarray_start array1 rs) in
  update_window_surface_rects w rs count

let update_window_surface_rects w rs =
  let count = List.length rs in
  let rs = to_voidp (CArray.start (CArray.of_list rect rs)) in
  update_window_surface_rects w rs count

(* OpenGL contexts *)

type gl_context = unit ptr
let gl_context : unit ptr typ = ptr void
let gl_context_opt : unit ptr option typ = ptr_opt void

let unsafe_gl_context_of_ptr addr : gl_context =
  ptr_of_raw_address addr
let unsafe_ptr_of_gl_context gl_context =
  raw_address_of_ptr (to_voidp gl_context)

module Gl = struct
  type context_flags = int
  let context_debug_flag = sdl_gl_context_debug_flag
  let context_forward_compatible_flag = sdl_gl_context_forward_compatible_flag
  let context_robust_access_flag = sdl_gl_context_robust_access_flag
  let context_reset_isolation_flag = sdl_gl_context_reset_isolation_flag

  type profile = int
  let context_profile_core = sdl_gl_context_profile_core
  let context_profile_compatibility = sdl_gl_context_profile_compatibility
  let context_profile_es = sdl_gl_context_profile_es

  type attr = int
  let red_size = sdl_gl_red_size
  let green_size = sdl_gl_green_size
  let blue_size = sdl_gl_blue_size
  let alpha_size = sdl_gl_alpha_size
  let buffer_size = sdl_gl_buffer_size
  let doublebuffer = sdl_gl_doublebuffer
  let depth_size = sdl_gl_depth_size
  let stencil_size = sdl_gl_stencil_size
  let accum_red_size = sdl_gl_accum_red_size
  let accum_green_size = sdl_gl_accum_green_size
  let accum_blue_size = sdl_gl_accum_blue_size
  let accum_alpha_size = sdl_gl_accum_alpha_size
  let stereo = sdl_gl_stereo
  let multisamplebuffers = sdl_gl_multisamplebuffers
  let multisamplesamples = sdl_gl_multisamplesamples
  let accelerated_visual = sdl_gl_accelerated_visual
  let context_major_version = sdl_gl_context_major_version
  let context_minor_version = sdl_gl_context_minor_version
  let context_egl = sdl_gl_context_egl
  let context_flags = sdl_gl_context_flags
  let context_profile_mask = sdl_gl_context_profile_mask
  let share_with_current_context = sdl_gl_share_with_current_context
  let framebuffer_srgb_capable = sdl_gl_framebuffer_srgb_capable
end

let gl_bind_texture =
  foreign "SDL_GL_BindTexture"
    (texture @-> ptr float @-> ptr float @-> returning zero_to_ok)

let gl_bind_texture t =
  let w = allocate float 0. in
  let h = allocate float 0. in
  match gl_bind_texture t w h with
  | Ok () -> Ok (!@ w, !@ h) | Error _ as e -> e

let gl_create_context =
  foreign "SDL_GL_CreateContext"
    (window @-> returning (some_to_ok gl_context_opt))

let gl_delete_context =
  foreign "SDL_GL_DeleteContext" (gl_context @-> returning void)

let gl_extension_supported =
  foreign "SDL_GL_ExtensionSupported" (string @-> returning bool)

let gl_get_attribute =
  foreign "SDL_GL_GetAttribute" (int @-> (ptr int) @-> returning int)

let gl_get_attribute att =
  let value = allocate int 0 in
  match gl_get_attribute att value with
  | 0 -> Ok (!@ value) | err -> error ()

let gl_get_current_context =
  foreign "SDL_GL_GetCurrentContext"
    (void @-> returning (some_to_ok gl_context_opt))

let gl_get_drawable_size =
  foreign "SDL_GL_GetDrawableSize"
    (window @-> ptr int @-> ptr int @-> returning void)

let gl_get_drawable_size win =
  let w = allocate int 0 in
  let h = allocate int 0 in
  gl_get_drawable_size win w h;
  (!@ w, !@ h)

let gl_get_swap_interval =
  foreign "SDL_GL_GetSwapInterval" (void @-> returning nat_to_ok)

let gl_make_current =
  foreign "SDL_GL_MakeCurrent"
    (window @-> gl_context @-> returning zero_to_ok)

let gl_reset_attributes =
  foreign "SDL_GL_ResetAttributes" ~stub (void @-> returning void)

let gl_set_attribute =
  foreign "SDL_GL_SetAttribute" (int @-> int @-> returning zero_to_ok)

let gl_set_swap_interval =
  foreign "SDL_GL_SetSwapInterval" (int @-> returning zero_to_ok)

let gl_swap_window =
  foreign "SDL_GL_SwapWindow" (window @-> returning void)

let gl_unbind_texture =
  foreign "SDL_GL_UnbindTexture" (texture @-> returning zero_to_ok)

(* Screen saver *)

let disable_screen_saver =
  foreign "SDL_DisableScreenSaver" (void @-> returning void)

let enable_screen_saver =
  foreign "SDL_EnableScreenSaver" (void @-> returning void)

let is_screen_saver_enabled =
  foreign "SDL_IsScreenSaverEnabled" (void @-> returning bool)

(* Message boxes *)

module Message_box = struct
  let i = Unsigned.UInt32.of_int

  type button_flags = Unsigned.uint32
  let button_returnkey_default = i sdl_messagebox_button_returnkey_default
  let button_escapekey_default = i sdl_messagebox_button_escapekey_default

  type button_data =
    { button_flags : button_flags;
      button_id : int;
      button_text : string }

  let button_data = structure "SDL_MessageBoxButtonData"
  let button_flags = field button_data "flags" uint32_t
  let button_buttonid = field button_data "buttonid" int
  let button_text = field button_data "text" string
  let () = seal button_data

  type flags = Unsigned.uint32
  let error = i sdl_messagebox_error
  let warning = i sdl_messagebox_warning
  let information = i sdl_messagebox_information

  type color = int * int * int
  let color = structure "SDL_MessageBoxColor"
  let color_r = field color "r" uint8_t
  let color_g = field color "g" uint8_t
  let color_b = field color "b" uint8_t
  let () = seal color

  type color_type = int
  let color_background = sdl_messagebox_color_background
  let color_text = sdl_messagebox_color_text
  let color_button_border = sdl_messagebox_color_button_border
  let color_button_background = sdl_messagebox_color_button_background
  let color_button_selected = sdl_messagebox_color_button_selected
  let color_button_max = sdl_messagebox_color_max

  type color_scheme =
    { color_background : color;
      color_text : color;
      color_button_border : color;
      color_button_background : color;
      color_button_selected : color; }

  let color_scheme = structure "SDL_MessageBoxColorScheme"
  let colors = field color_scheme "colors" (array color_button_max color)
  let () = seal color_scheme

  type data =
    { flags : flags;
      window : window option;
      title : string;
      message : string;
      buttons : button_data list;
      color_scheme : color_scheme option }

  let data = structure "SDL_MessageBoxData"
  let d_flags = field data "flags" uint32_t
  let d_window = field data "window" window
  let d_title = field data "title" string
  let d_message = field data "message" string
  let d_numbuttons = field data "numbuttons" int
  let d_buttons = field data "buttons" (ptr button_data)
  let d_color_scheme = field data "colorScheme" (ptr color_scheme)
  let () = seal data

  let buttons_to_c bl =
    let button_data_to_c b =
      let bt = make button_data in
      setf bt button_flags b.button_flags;
      setf bt button_buttonid b.button_id;
      setf bt button_text b.button_text;
      bt
    in
    CArray.start (CArray.of_list button_data (List.map button_data_to_c bl))

  let color_scheme_to_c s =
    let st = make color_scheme in
    let colors = getf st colors in
    let set i (rv, gv, bv) =
      let ct = CArray.get colors i in
      setf ct color_r (Unsigned.UInt8.of_int rv);
      setf ct color_g (Unsigned.UInt8.of_int rv);
      setf ct color_b (Unsigned.UInt8.of_int rv);
    in
    set color_background s.color_background;
    set color_text s.color_text;
    set color_button_border s.color_button_border;
    set color_button_background s.color_button_background;
    set color_button_selected s.color_button_selected;
    st

  let data_to_c d =
    let dt = make data in
    setf dt d_flags d.flags;
    setf dt d_window (match d.window with None -> null | Some w -> w);
    setf dt d_title d.title;
    setf dt d_message d.message;
    setf dt d_numbuttons (List.length d.buttons);
    setf dt d_buttons (buttons_to_c d.buttons);
    setf dt d_color_scheme
      begin match d.color_scheme with
      | None -> coerce (ptr void) (ptr color_scheme) null
      | Some s -> addr (color_scheme_to_c s)
      end;
    dt
end

let show_message_box =
  foreign "SDL_ShowMessageBox"
    (ptr Message_box.data @-> ptr int @-> returning zero_to_ok)

let show_message_box d =
  let d = addr (Message_box.data_to_c d) in
  let ret = allocate int 0 in
  match show_message_box d ret with
  | Ok () -> Ok (!@ ret) | Error _ as e -> e

let show_simple_message_box =
  foreign "SDL_ShowSimpleMessageBox"
    (uint32_t @-> string @-> string @-> window_opt @-> returning zero_to_ok)

let show_simple_message_box t ~title msg w =
  show_simple_message_box t title msg w

(* Clipboard *)

let get_clipboard_text =
  foreign "SDL_GetClipboardText" (void @-> returning (ptr char))

let get_clipboard_text () =
  let p = get_clipboard_text () in
  if (to_voidp p) = null then error () else
  let b = Buffer.create 255 in
  let ptr = ref p in
  while (!@ !ptr) <> '\000' do
    Buffer.add_char b (!@ !ptr);
    ptr := !ptr +@ 1;
  done;
  sdl_free (to_voidp p);
  Ok (Buffer.contents b)

let has_clipboard_text =
  foreign "SDL_HasClipboardText" (void @-> returning bool)

let set_clipboard_text =
  foreign "SDL_SetClipboardText" (string @-> returning zero_to_ok)

(* Input *)

type button_state = uint8
let pressed = sdl_pressed
let released = sdl_released

type toggle_state = uint8
let disable = sdl_disable
let enable = sdl_enable

(* Keyboard *)

type scancode = int
let scancode = int

module Scancode = struct
  let num_scancodes = sdl_num_scancodes
  let unknown = sdl_scancode_unknown
  let a = sdl_scancode_a
  let b = sdl_scancode_b
  let c = sdl_scancode_c
  let d = sdl_scancode_d
  let e = sdl_scancode_e
  let f = sdl_scancode_f
  let g = sdl_scancode_g
  let h = sdl_scancode_h
  let i = sdl_scancode_i
  let j = sdl_scancode_j
  let k = sdl_scancode_k
  let l = sdl_scancode_l
  let m = sdl_scancode_m
  let n = sdl_scancode_n
  let o = sdl_scancode_o
  let p = sdl_scancode_p
  let q = sdl_scancode_q
  let r = sdl_scancode_r
  let s = sdl_scancode_s
  let t = sdl_scancode_t
  let u = sdl_scancode_u
  let v = sdl_scancode_v
  let w = sdl_scancode_w
  let x = sdl_scancode_x
  let y = sdl_scancode_y
  let z = sdl_scancode_z
  let k1 = sdl_scancode_1
  let k2 = sdl_scancode_2
  let k3 = sdl_scancode_3
  let k4 = sdl_scancode_4
  let k5 = sdl_scancode_5
  let k6 = sdl_scancode_6
  let k7 = sdl_scancode_7
  let k8 = sdl_scancode_8
  let k9 = sdl_scancode_9
  let k0 = sdl_scancode_0
  let return = sdl_scancode_return
  let escape = sdl_scancode_escape
  let backspace = sdl_scancode_backspace
  let tab = sdl_scancode_tab
  let space = sdl_scancode_space
  let minus = sdl_scancode_minus
  let equals = sdl_scancode_equals
  let leftbracket = sdl_scancode_leftbracket
  let rightbracket = sdl_scancode_rightbracket
  let backslash = sdl_scancode_backslash
  let nonushash = sdl_scancode_nonushash
  let semicolon = sdl_scancode_semicolon
  let apostrophe = sdl_scancode_apostrophe
  let grave = sdl_scancode_grave
  let comma = sdl_scancode_comma
  let period = sdl_scancode_period
  let slash = sdl_scancode_slash
  let capslock = sdl_scancode_capslock
  let f1 = sdl_scancode_f1
  let f2 = sdl_scancode_f2
  let f3 = sdl_scancode_f3
  let f4 = sdl_scancode_f4
  let f5 = sdl_scancode_f5
  let f6 = sdl_scancode_f6
  let f7 = sdl_scancode_f7
  let f8 = sdl_scancode_f8
  let f9 = sdl_scancode_f9
  let f10 = sdl_scancode_f10
  let f11 = sdl_scancode_f11
  let f12 = sdl_scancode_f12
  let printscreen = sdl_scancode_printscreen
  let scrolllock = sdl_scancode_scrolllock
  let pause = sdl_scancode_pause
  let insert = sdl_scancode_insert
  let home = sdl_scancode_home
  let pageup = sdl_scancode_pageup
  let delete = sdl_scancode_delete
  let kend = sdl_scancode_end
  let pagedown = sdl_scancode_pagedown
  let right = sdl_scancode_right
  let left = sdl_scancode_left
  let down = sdl_scancode_down
  let up = sdl_scancode_up
  let numlockclear = sdl_scancode_numlockclear
  let kp_divide = sdl_scancode_kp_divide
  let kp_multiply = sdl_scancode_kp_multiply
  let kp_minus = sdl_scancode_kp_minus
  let kp_plus = sdl_scancode_kp_plus
  let kp_enter = sdl_scancode_kp_enter
  let kp_1 = sdl_scancode_kp_1
  let kp_2 = sdl_scancode_kp_2
  let kp_3 = sdl_scancode_kp_3
  let kp_4 = sdl_scancode_kp_4
  let kp_5 = sdl_scancode_kp_5
  let kp_6 = sdl_scancode_kp_6
  let kp_7 = sdl_scancode_kp_7
  let kp_8 = sdl_scancode_kp_8
  let kp_9 = sdl_scancode_kp_9
  let kp_0 = sdl_scancode_kp_0
  let kp_period = sdl_scancode_kp_period
  let nonusbackslash = sdl_scancode_nonusbackslash
  let application = sdl_scancode_application
  let kp_equals = sdl_scancode_kp_equals
  let f13 = sdl_scancode_f13
  let f14 = sdl_scancode_f14
  let f15 = sdl_scancode_f15
  let f16 = sdl_scancode_f16
  let f17 = sdl_scancode_f17
  let f18 = sdl_scancode_f18
  let f19 = sdl_scancode_f19
  let f20 = sdl_scancode_f20
  let f21 = sdl_scancode_f21
  let f22 = sdl_scancode_f22
  let f23 = sdl_scancode_f23
  let f24 = sdl_scancode_f24
  let execute = sdl_scancode_execute
  let help = sdl_scancode_help
  let menu = sdl_scancode_menu
  let select = sdl_scancode_select
  let stop = sdl_scancode_stop
  let again = sdl_scancode_again
  let undo = sdl_scancode_undo
  let cut = sdl_scancode_cut
  let copy = sdl_scancode_copy
  let paste = sdl_scancode_paste
  let find = sdl_scancode_find
  let mute = sdl_scancode_mute
  let volumeup = sdl_scancode_volumeup
  let volumedown = sdl_scancode_volumedown
  let kp_comma = sdl_scancode_kp_comma
  let kp_equalsas400 = sdl_scancode_kp_equalsas400
  let international1 = sdl_scancode_international1
  let international2 = sdl_scancode_international2
  let international3 = sdl_scancode_international3
  let international4 = sdl_scancode_international4
  let international5 = sdl_scancode_international5
  let international6 = sdl_scancode_international6
  let international7 = sdl_scancode_international7
  let international8 = sdl_scancode_international8
  let international9 = sdl_scancode_international9
  let lang1 = sdl_scancode_lang1
  let lang2 = sdl_scancode_lang2
  let lang3 = sdl_scancode_lang3
  let lang4 = sdl_scancode_lang4
  let lang5 = sdl_scancode_lang5
  let lang6 = sdl_scancode_lang6
  let lang7 = sdl_scancode_lang7
  let lang8 = sdl_scancode_lang8
  let lang9 = sdl_scancode_lang9
  let alterase = sdl_scancode_alterase
  let sysreq = sdl_scancode_sysreq
  let cancel = sdl_scancode_cancel
  let clear = sdl_scancode_clear
  let prior = sdl_scancode_prior
  let return2 = sdl_scancode_return2
  let separator = sdl_scancode_separator
  let out = sdl_scancode_out
  let oper = sdl_scancode_oper
  let clearagain = sdl_scancode_clearagain
  let crsel = sdl_scancode_crsel
  let exsel = sdl_scancode_exsel
  let kp_00 = sdl_scancode_kp_00
  let kp_000 = sdl_scancode_kp_000
  let thousandsseparator = sdl_scancode_thousandsseparator
  let decimalseparator = sdl_scancode_decimalseparator
  let currencyunit = sdl_scancode_currencyunit
  let currencysubunit = sdl_scancode_currencysubunit
  let kp_leftparen = sdl_scancode_kp_leftparen
  let kp_rightparen = sdl_scancode_kp_rightparen
  let kp_leftbrace = sdl_scancode_kp_leftbrace
  let kp_rightbrace = sdl_scancode_kp_rightbrace
  let kp_tab = sdl_scancode_kp_tab
  let kp_backspace = sdl_scancode_kp_backspace
  let kp_a = sdl_scancode_kp_a
  let kp_b = sdl_scancode_kp_b
  let kp_c = sdl_scancode_kp_c
  let kp_d = sdl_scancode_kp_d
  let kp_e = sdl_scancode_kp_e
  let kp_f = sdl_scancode_kp_f
  let kp_xor = sdl_scancode_kp_xor
  let kp_power = sdl_scancode_kp_power
  let kp_percent = sdl_scancode_kp_percent
  let kp_less = sdl_scancode_kp_less
  let kp_greater = sdl_scancode_kp_greater
  let kp_ampersand = sdl_scancode_kp_ampersand
  let kp_dblampersand = sdl_scancode_kp_dblampersand
  let kp_verticalbar = sdl_scancode_kp_verticalbar
  let kp_dblverticalbar = sdl_scancode_kp_dblverticalbar
  let kp_colon = sdl_scancode_kp_colon
  let kp_hash = sdl_scancode_kp_hash
  let kp_space = sdl_scancode_kp_space
  let kp_at = sdl_scancode_kp_at
  let kp_exclam = sdl_scancode_kp_exclam
  let kp_memstore = sdl_scancode_kp_memstore
  let kp_memrecall = sdl_scancode_kp_memrecall
  let kp_memclear = sdl_scancode_kp_memclear
  let kp_memadd = sdl_scancode_kp_memadd
  let kp_memsubtract = sdl_scancode_kp_memsubtract
  let kp_memmultiply = sdl_scancode_kp_memmultiply
  let kp_memdivide = sdl_scancode_kp_memdivide
  let kp_plusminus = sdl_scancode_kp_plusminus
  let kp_clear = sdl_scancode_kp_clear
  let kp_clearentry = sdl_scancode_kp_clearentry
  let kp_binary = sdl_scancode_kp_binary
  let kp_octal = sdl_scancode_kp_octal
  let kp_decimal = sdl_scancode_kp_decimal
  let kp_hexadecimal = sdl_scancode_kp_hexadecimal
  let lctrl = sdl_scancode_lctrl
  let lshift = sdl_scancode_lshift
  let lalt = sdl_scancode_lalt
  let lgui = sdl_scancode_lgui
  let rctrl = sdl_scancode_rctrl
  let rshift = sdl_scancode_rshift
  let ralt = sdl_scancode_ralt
  let rgui = sdl_scancode_rgui
  let mode = sdl_scancode_mode
  let audionext = sdl_scancode_audionext
  let audioprev = sdl_scancode_audioprev
  let audiostop = sdl_scancode_audiostop
  let audioplay = sdl_scancode_audioplay
  let audiomute = sdl_scancode_audiomute
  let mediaselect = sdl_scancode_mediaselect
  let www = sdl_scancode_www
  let mail = sdl_scancode_mail
  let calculator = sdl_scancode_calculator
  let computer = sdl_scancode_computer
  let ac_search = sdl_scancode_ac_search
  let ac_home = sdl_scancode_ac_home
  let ac_back = sdl_scancode_ac_back
  let ac_forward = sdl_scancode_ac_forward
  let ac_stop = sdl_scancode_ac_stop
  let ac_refresh = sdl_scancode_ac_refresh
  let ac_bookmarks = sdl_scancode_ac_bookmarks
  let brightnessdown = sdl_scancode_brightnessdown
  let brightnessup = sdl_scancode_brightnessup
  let displayswitch = sdl_scancode_displayswitch
  let kbdillumtoggle = sdl_scancode_kbdillumtoggle
  let kbdillumdown = sdl_scancode_kbdillumdown
  let kbdillumup = sdl_scancode_kbdillumup
  let eject = sdl_scancode_eject
  let sleep = sdl_scancode_sleep
  let app1 = sdl_scancode_app1
  let app2 = sdl_scancode_app2

  let enum_of_scancode = [|
    `Unknown; `Unknown; `Unknown; `Unknown; `A; `B; `C; `D; `E; `F;
    `G; `H; `I; `J; `K; `L; `M; `N; `O; `P; `Q; `R; `S; `T; `U; `V;
    `W; `X; `Y; `Z; `K1; `K2; `K3; `K4; `K5; `K6; `K7; `K8; `K9; `K0;
    `Return; `Escape; `Backspace; `Tab; `Space; `Minus; `Equals;
    `Leftbracket; `Rightbracket; `Backslash; `Nonushash; `Semicolon;
    `Apostrophe; `Grave; `Comma; `Period; `Slash; `Capslock; `F1; `F2;
    `F3; `F4; `F5; `F6; `F7; `F8; `F9; `F10; `F11; `F12; `Printscreen;
    `Scrolllock; `Pause; `Insert; `Home; `Pageup; `Delete; `End;
    `Pagedown; `Right; `Left; `Down; `Up; `Numlockclear; `Kp_divide;
    `Kp_multiply; `Kp_minus; `Kp_plus; `Kp_enter; `Kp_1; `Kp_2; `Kp_3;
    `Kp_4; `Kp_5; `Kp_6; `Kp_7; `Kp_8; `Kp_9; `Kp_0; `Kp_period;
    `Nonusbackslash; `Application; `Power; `Kp_equals; `F13; `F14;
    `F15; `F16; `F17; `F18; `F19; `F20; `F21; `F22; `F23; `F24;
    `Execute; `Help; `Menu; `Select; `Stop; `Again; `Undo; `Cut;
    `Copy; `Paste; `Find; `Mute; `Volumeup; `Volumedown; `Unknown;
    `Unknown; `Unknown; `Kp_comma; `Kp_equalsas400; `International1;
    `International2; `International3; `International4;
    `International5; `International6; `International7;
    `International8; `International9; `Lang1; `Lang2; `Lang3; `Lang4;
    `Lang5; `Lang6; `Lang7; `Lang8; `Lang9; `Alterase; `Sysreq;
    `Cancel; `Clear; `Prior; `Return2; `Separator; `Out; `Oper;
    `Clearagain; `Crsel; `Exsel; `Unknown; `Unknown; `Unknown;
    `Unknown; `Unknown; `Unknown; `Unknown; `Unknown; `Unknown;
    `Unknown; `Unknown; `Kp_00; `Kp_000; `Thousandsseparator;
    `Decimalseparator; `Currencyunit; `Currencysubunit; `Kp_leftparen;
    `Kp_rightparen; `Kp_leftbrace; `Kp_rightbrace; `Kp_tab;
    `Kp_backspace; `Kp_a; `Kp_b; `Kp_c; `Kp_d; `Kp_e; `Kp_f; `Kp_xor;
    `Kp_power; `Kp_percent; `Kp_less; `Kp_greater; `Kp_ampersand;
    `Kp_dblampersand; `Kp_verticalbar; `Kp_dblverticalbar; `Kp_colon;
    `Kp_hash; `Kp_space; `Kp_at; `Kp_exclam; `Kp_memstore;
    `Kp_memrecall; `Kp_memclear; `Kp_memadd; `Kp_memsubtract;
    `Kp_memmultiply; `Kp_memdivide; `Kp_plusminus; `Kp_clear;
    `Kp_clearentry; `Kp_binary; `Kp_octal; `Kp_decimal;
    `Kp_hexadecimal; `Unknown; `Unknown; `Lctrl; `Lshift; `Lalt;
    `Lgui; `Rctrl; `Rshift; `Ralt; `Rgui; `Unknown; `Unknown;
    `Unknown; `Unknown; `Unknown; `Unknown; `Unknown; `Unknown;
    `Unknown; `Unknown; `Unknown; `Unknown; `Unknown; `Unknown;
    `Unknown; `Unknown; `Unknown; `Unknown; `Unknown; `Unknown;
    `Unknown; `Unknown; `Unknown; `Unknown; `Unknown; `Mode;
    `Audionext; `Audioprev; `Audiostop; `Audioplay; `Audiomute;
    `Mediaselect; `Www; `Mail; `Calculator; `Computer; `Ac_search;
    `Ac_home; `Ac_back; `Ac_forward; `Ac_stop; `Ac_refresh;
    `Ac_bookmarks; `Brightnessdown; `Brightnessup; `Displayswitch;
    `Kbdillumtoggle; `Kbdillumdown; `Kbdillumup; `Eject; `Sleep;
    `App1; `App2; |]

  let enum s =
    if 0 <= s && s <= app2 then unsafe_get enum_of_scancode s else
    `Unknown
end

type keycode = int
let keycode = int

module K = struct
  let scancode_mask = sdlk_scancode_mask
  let unknown = sdlk_unknown
  let return = sdlk_return
  let escape = sdlk_escape
  let backspace = sdlk_backspace
  let tab = sdlk_tab
  let space = sdlk_space
  let exclaim = sdlk_exclaim
  let quotedbl = sdlk_quotedbl
  let hash = sdlk_hash
  let percent = sdlk_percent
  let dollar = sdlk_dollar
  let ampersand = sdlk_ampersand
  let quote = sdlk_quote
  let leftparen = sdlk_leftparen
  let rightparen = sdlk_rightparen
  let asterisk = sdlk_asterisk
  let plus = sdlk_plus
  let comma = sdlk_comma
  let minus = sdlk_minus
  let period = sdlk_period
  let slash = sdlk_slash
  let k0 = sdlk_0
  let k1 = sdlk_1
  let k2 = sdlk_2
  let k3 = sdlk_3
  let k4 = sdlk_4
  let k5 = sdlk_5
  let k6 = sdlk_6
  let k7 = sdlk_7
  let k8 = sdlk_8
  let k9 = sdlk_9
  let colon = sdlk_colon
  let semicolon = sdlk_semicolon
  let less = sdlk_less
  let equals = sdlk_equals
  let greater = sdlk_greater
  let question = sdlk_question
  let at = sdlk_at
  let leftbracket = sdlk_leftbracket
  let backslash = sdlk_backslash
  let rightbracket = sdlk_rightbracket
  let caret = sdlk_caret
  let underscore = sdlk_underscore
  let backquote = sdlk_backquote
  let a = sdlk_a
  let b = sdlk_b
  let c = sdlk_c
  let d = sdlk_d
  let e = sdlk_e
  let f = sdlk_f
  let g = sdlk_g
  let h = sdlk_h
  let i = sdlk_i
  let j = sdlk_j
  let k = sdlk_k
  let l = sdlk_l
  let m = sdlk_m
  let n = sdlk_n
  let o = sdlk_o
  let p = sdlk_p
  let q = sdlk_q
  let r = sdlk_r
  let s = sdlk_s
  let t = sdlk_t
  let u = sdlk_u
  let v = sdlk_v
  let w = sdlk_w
  let x = sdlk_x
  let y = sdlk_y
  let z = sdlk_z
  let capslock = sdlk_capslock
  let f1 = sdlk_f1
  let f2 = sdlk_f2
  let f3 = sdlk_f3
  let f4 = sdlk_f4
  let f5 = sdlk_f5
  let f6 = sdlk_f6
  let f7 = sdlk_f7
  let f8 = sdlk_f8
  let f9 = sdlk_f9
  let f10 = sdlk_f10
  let f11 = sdlk_f11
  let f12 = sdlk_f12
  let printscreen = sdlk_printscreen
  let scrolllock = sdlk_scrolllock
  let pause = sdlk_pause
  let insert = sdlk_insert
  let home = sdlk_home
  let pageup = sdlk_pageup
  let delete = sdlk_delete
  let kend = sdlk_end
  let pagedown = sdlk_pagedown
  let right = sdlk_right
  let left = sdlk_left
  let down = sdlk_down
  let up = sdlk_up
  let numlockclear = sdlk_numlockclear
  let kp_divide = sdlk_kp_divide
  let kp_multiply = sdlk_kp_multiply
  let kp_minus = sdlk_kp_minus
  let kp_plus = sdlk_kp_plus
  let kp_enter = sdlk_kp_enter
  let kp_1 = sdlk_kp_1
  let kp_2 = sdlk_kp_2
  let kp_3 = sdlk_kp_3
  let kp_4 = sdlk_kp_4
  let kp_5 = sdlk_kp_5
  let kp_6 = sdlk_kp_6
  let kp_7 = sdlk_kp_7
  let kp_8 = sdlk_kp_8
  let kp_9 = sdlk_kp_9
  let kp_0 = sdlk_kp_0
  let kp_period = sdlk_kp_period
  let application = sdlk_application
  let power = sdlk_power
  let kp_equals = sdlk_kp_equals
  let f13 = sdlk_f13
  let f14 = sdlk_f14
  let f15 = sdlk_f15
  let f16 = sdlk_f16
  let f17 = sdlk_f17
  let f18 = sdlk_f18
  let f19 = sdlk_f19
  let f20 = sdlk_f20
  let f21 = sdlk_f21
  let f22 = sdlk_f22
  let f23 = sdlk_f23
  let f24 = sdlk_f24
  let execute = sdlk_execute
  let help = sdlk_help
  let menu = sdlk_menu
  let select = sdlk_select
  let stop = sdlk_stop
  let again = sdlk_again
  let undo = sdlk_undo
  let cut = sdlk_cut
  let copy = sdlk_copy
  let paste = sdlk_paste
  let find = sdlk_find
  let mute = sdlk_mute
  let volumeup = sdlk_volumeup
  let volumedown = sdlk_volumedown
  let kp_comma = sdlk_kp_comma
  let kp_equalsas400 = sdlk_kp_equalsas400
  let alterase = sdlk_alterase
  let sysreq = sdlk_sysreq
  let cancel = sdlk_cancel
  let clear = sdlk_clear
  let prior = sdlk_prior
  let return2 = sdlk_return2
  let separator = sdlk_separator
  let out = sdlk_out
  let oper = sdlk_oper
  let clearagain = sdlk_clearagain
  let crsel = sdlk_crsel
  let exsel = sdlk_exsel
  let kp_00 = sdlk_kp_00
  let kp_000 = sdlk_kp_000
  let thousandsseparator = sdlk_thousandsseparator
  let decimalseparator = sdlk_decimalseparator
  let currencyunit = sdlk_currencyunit
  let currencysubunit = sdlk_currencysubunit
  let kp_leftparen = sdlk_kp_leftparen
  let kp_rightparen = sdlk_kp_rightparen
  let kp_leftbrace = sdlk_kp_leftbrace
  let kp_rightbrace = sdlk_kp_rightbrace
  let kp_tab = sdlk_kp_tab
  let kp_backspace = sdlk_kp_backspace
  let kp_a = sdlk_kp_a
  let kp_b = sdlk_kp_b
  let kp_c = sdlk_kp_c
  let kp_d = sdlk_kp_d
  let kp_e = sdlk_kp_e
  let kp_f = sdlk_kp_f
  let kp_xor = sdlk_kp_xor
  let kp_power = sdlk_kp_power
  let kp_percent = sdlk_kp_percent
  let kp_less = sdlk_kp_less
  let kp_greater = sdlk_kp_greater
  let kp_ampersand = sdlk_kp_ampersand
  let kp_dblampersand = sdlk_kp_dblampersand
  let kp_verticalbar = sdlk_kp_verticalbar
  let kp_dblverticalbar = sdlk_kp_dblverticalbar
  let kp_colon = sdlk_kp_colon
  let kp_hash = sdlk_kp_hash
  let kp_space = sdlk_kp_space
  let kp_at = sdlk_kp_at
  let kp_exclam = sdlk_kp_exclam
  let kp_memstore = sdlk_kp_memstore
  let kp_memrecall = sdlk_kp_memrecall
  let kp_memclear = sdlk_kp_memclear
  let kp_memadd = sdlk_kp_memadd
  let kp_memsubtract = sdlk_kp_memsubtract
  let kp_memmultiply = sdlk_kp_memmultiply
  let kp_memdivide = sdlk_kp_memdivide
  let kp_plusminus = sdlk_kp_plusminus
  let kp_clear = sdlk_kp_clear
  let kp_clearentry = sdlk_kp_clearentry
  let kp_binary = sdlk_kp_binary
  let kp_octal = sdlk_kp_octal
  let kp_decimal = sdlk_kp_decimal
  let kp_hexadecimal = sdlk_kp_hexadecimal
  let lctrl = sdlk_lctrl
  let lshift = sdlk_lshift
  let lalt = sdlk_lalt
  let lgui = sdlk_lgui
  let rctrl = sdlk_rctrl
  let rshift = sdlk_rshift
  let ralt = sdlk_ralt
  let rgui = sdlk_rgui
  let mode = sdlk_mode
  let audionext = sdlk_audionext
  let audioprev = sdlk_audioprev
  let audiostop = sdlk_audiostop
  let audioplay = sdlk_audioplay
  let audiomute = sdlk_audiomute
  let mediaselect = sdlk_mediaselect
  let www = sdlk_www
  let mail = sdlk_mail
  let calculator = sdlk_calculator
  let computer = sdlk_computer
  let ac_search = sdlk_ac_search
  let ac_home = sdlk_ac_home
  let ac_back = sdlk_ac_back
  let ac_forward = sdlk_ac_forward
  let ac_stop = sdlk_ac_stop
  let ac_refresh = sdlk_ac_refresh
  let ac_bookmarks = sdlk_ac_bookmarks
  let brightnessdown = sdlk_brightnessdown
  let brightnessup = sdlk_brightnessup
  let displayswitch = sdlk_displayswitch
  let kbdillumtoggle = sdlk_kbdillumtoggle
  let kbdillumdown = sdlk_kbdillumdown
  let kbdillumup = sdlk_kbdillumup
  let eject = sdlk_eject
  let sleep = sdlk_sleep
end

type keymod = int
let keymod = int_as_uint16_t

module Kmod = struct
  let none = kmod_none
  let lshift = kmod_lshift
  let rshift = kmod_rshift
  let lctrl = kmod_lctrl
  let rctrl = kmod_rctrl
  let lalt = kmod_lalt
  let ralt = kmod_ralt
  let lgui = kmod_lgui
  let rgui = kmod_rgui
  let num = kmod_num
  let caps = kmod_caps
  let mode = kmod_mode
  let reserved = kmod_reserved
  let ctrl = kmod_ctrl
  let shift = kmod_shift
  let alt = kmod_alt
  let gui = kmod_gui
end

let get_keyboard_focus =
  foreign "SDL_GetKeyboardFocus" (void @-> returning window_opt)

let get_keyboard_state =
  foreign "SDL_GetKeyboardState" (ptr int @-> returning (ptr int))

let get_keyboard_state () =
  let count = allocate int 0 in
  let p = get_keyboard_state count in
  bigarray_of_ptr array1 (!@ count) Bigarray.int8_unsigned p

let get_key_from_name =
  foreign "SDL_GetKeyFromName" (string @-> returning keycode)

let get_key_from_scancode =
  foreign "SDL_GetKeyFromScancode" (scancode @-> returning keycode)

let get_key_name =
  foreign "SDL_GetKeyName" (keycode @-> returning string)

let get_mod_state =
  foreign "SDL_GetModState" (void @-> returning keymod)

let get_scancode_from_key =
  foreign "SDL_GetScancodeFromKey" (keycode @-> returning scancode)

let get_scancode_from_name =
  foreign "SDL_GetScancodeFromName" (string @-> returning scancode)

let get_scancode_name =
  foreign "SDL_GetScancodeName" (scancode @-> returning string)

let has_screen_keyboard_support =
  foreign "SDL_HasScreenKeyboardSupport" (void @-> returning bool)

let is_screen_keyboard_shown =
  foreign "SDL_IsScreenKeyboardShown" (window @-> returning bool)

let is_text_input_active =
  foreign "SDL_IsTextInputActive" (void @-> returning bool)

let set_mod_state =
  foreign "SDL_SetModState" (keymod @-> returning void)

let set_text_input_rect =
  foreign "SDL_SetTextInputRect" (ptr rect @-> returning void)

let set_text_input_rect r =
  set_text_input_rect (Rect.opt_addr r)

let start_text_input =
  foreign "SDL_StartTextInput" (void @-> returning void)

let stop_text_input =
  foreign "SDL_StopTextInput" (void @-> returning void)

(* Mouse *)

type cursor = unit ptr
let cursor : cursor typ = ptr void
let cursor_opt : cursor option typ = ptr_opt void

let unsafe_cursor_of_ptr addr : cursor =
  ptr_of_raw_address addr
let unsafe_ptr_of_cursor cursor =
  raw_address_of_ptr (to_voidp cursor)

module System_cursor = struct
  type t = int
  let arrow = sdl_system_cursor_arrow
  let ibeam = sdl_system_cursor_ibeam
  let wait = sdl_system_cursor_wait
  let crosshair = sdl_system_cursor_crosshair
  let waitarrow = sdl_system_cursor_waitarrow
  let size_nw_se = sdl_system_cursor_sizenwse
  let size_ne_sw = sdl_system_cursor_sizenesw
  let size_we = sdl_system_cursor_sizewe
  let size_ns = sdl_system_cursor_sizens
  let size_all = sdl_system_cursor_sizeall
  let no = sdl_system_cursor_no
  let hand = sdl_system_cursor_hand
end

module Button = struct
  let left = sdl_button_left
  let right = sdl_button_right
  let middle = sdl_button_middle
  let x1 = sdl_button_x1
  let x2 = sdl_button_x2

  let i = Int32.of_int
  let lmask = i sdl_button_lmask
  let mmask = i sdl_button_mmask
  let rmask = i sdl_button_rmask
  let x1mask = i sdl_button_x1mask
  let x2mask = i sdl_button_x2mask
end

let create_color_cursor =
  foreign "SDL_CreateColorCursor"
    (surface @-> int @-> int @-> returning (some_to_ok cursor_opt))

let create_color_cursor s ~hot_x ~hot_y =
  create_color_cursor s hot_x hot_y

let create_cursor =
  foreign "SDL_CreateCursor"
    (ptr void @-> ptr void @-> int @-> int @-> int @-> int @->
     returning (some_to_ok cursor_opt))

let create_cursor d m ~w ~h ~hot_x ~hot_y =
  (* FIXME: we could try to check bounds *)
  let d = to_voidp (bigarray_start array1 d) in
  let m = to_voidp (bigarray_start array1 m) in
  create_cursor d m w h hot_x hot_y

let create_system_cursor =
  foreign "SDL_CreateSystemCursor"
    (int @-> returning (some_to_ok cursor_opt))

let free_cursor =
  foreign "SDL_FreeCursor" (cursor @-> returning void)

let get_cursor =
  foreign "SDL_GetCursor" (void @-> returning cursor_opt)

let get_default_cursor =
  foreign "SDL_GetDefaultCursor" (void @-> returning cursor_opt)

let get_mouse_focus =
  foreign "SDL_GetMouseFocus" (void @-> returning window_opt)

let get_mouse_state =
  foreign "SDL_GetMouseState"
    (ptr int @-> ptr int @-> returning int32_as_uint32_t)

let get_mouse_state () =
  let x = allocate int 0 in
  let y = allocate int 0 in
  let s = get_mouse_state x y in
  s, (!@ x, !@ y)

let get_relative_mouse_mode =
  foreign "SDL_GetRelativeMouseMode" (void @-> returning bool)

let get_relative_mouse_state =
  foreign "SDL_GetRelativeMouseState"
    (ptr int @-> ptr int @-> returning int32_as_uint32_t)

let get_relative_mouse_state () =
  let x = allocate int 0 in
  let y = allocate int 0 in
  let s = get_relative_mouse_state x y in
  s, (!@ x, !@ y)

let show_cursor =
  foreign "SDL_ShowCursor" (int @-> returning bool_to_ok)

let get_cursor_shown () =
  show_cursor (-1)

let set_cursor =
  foreign "SDL_SetCursor" (cursor_opt @-> returning void)

let set_relative_mouse_mode =
  foreign "SDL_SetRelativeMouseMode" (bool @-> returning zero_to_ok)

let show_cursor b =
  show_cursor (if b then 1 else 0)

let warp_mouse_in_window =
  foreign "SDL_WarpMouseInWindow"
    (window_opt @-> int @-> int @-> returning void)

let warp_mouse_in_window w ~x ~y =
  warp_mouse_in_window w x y

(* Touch *)

type touch_id = int64
let touch_id = int64_t
let touch_mouse_id = Int64.of_int32 (sdl_touch_mouseid)

type gesture_id = int64
let gesture_id = int64_t

type finger_id = int64
let finger_id = int64_t

type _finger
type finger = _finger structure
let finger : finger typ = structure "SDL_Finger"
let finger_finger_id = field finger "id" finger_id
let finger_x = field finger "x" float
let finger_y = field finger "y" float
let finger_pressure = field finger "pressure" float
let () = seal finger

module Finger = struct
  let id f = getf f finger_finger_id
  let x f = getf f finger_x
  let y f = getf f finger_y
  let pressure f = getf f finger_pressure
end

let get_num_touch_devices =
  foreign "SDL_GetNumTouchDevices" (void @-> returning int)

let get_num_touch_fingers =
  foreign "SDL_GetNumTouchFingers" (touch_id @-> returning int)

let get_touch_device =
  foreign "SDL_GetTouchDevice" (int @-> returning touch_id)

let get_touch_device i =
  match get_touch_device i with
  | 0L -> error () | id -> Ok id

let get_touch_finger =
  foreign "SDL_GetTouchFinger"
    (touch_id @-> int @-> returning (ptr_opt finger))

let get_touch_finger id i =
  match get_touch_finger id i with
  | None -> None | Some p -> Some (!@ p)

let load_dollar_templates =
  foreign "SDL_LoadDollarTemplates"
    (touch_id @-> rw_ops @-> returning zero_to_ok)

let record_gesture =
  foreign "SDL_RecordGesture" (touch_id @-> returning one_to_ok)

let save_dollar_template =
  foreign "SDL_SaveDollarTemplate"
    (gesture_id @-> rw_ops @-> returning zero_to_ok)

let save_all_dollar_templates =
  foreign "SDL_SaveAllDollarTemplates" (rw_ops @-> returning zero_to_ok)

(* Joystick *)

type _joystick_guid
type joystick_guid = _joystick_guid structure
let joystick_guid : joystick_guid typ = structure "SDL_JoystickGUID"
(* FIXME: No array here, see
   https://github.com/ocamllabs/ocaml-ctypes/issues/113 *)
(* let _= field joystick_guid "data" (array 16 uint8_t) *)
let _= field joystick_guid "data0" uint8_t
let _= field joystick_guid "data1" uint8_t
let _= field joystick_guid "data2" uint8_t
let _= field joystick_guid "data3" uint8_t
let _= field joystick_guid "data4" uint8_t
let _= field joystick_guid "data5" uint8_t
let _= field joystick_guid "data6" uint8_t
let _= field joystick_guid "data7" uint8_t
let _= field joystick_guid "data8" uint8_t
let _= field joystick_guid "data9" uint8_t
let _= field joystick_guid "data10" uint8_t
let _= field joystick_guid "data11" uint8_t
let _= field joystick_guid "data12" uint8_t
let _= field joystick_guid "data13" uint8_t
let _= field joystick_guid "data14" uint8_t
let _= field joystick_guid "data15" uint8_t
let () = seal joystick_guid

type joystick_id = int32
let joystick_id = int32_t

type joystick = unit ptr
let joystick : joystick typ = ptr void
let joystick_opt : joystick option typ = ptr_opt void

let unsafe_joystick_of_ptr addr : joystick =
  ptr_of_raw_address addr
let unsafe_ptr_of_joystick joystick =
  raw_address_of_ptr (to_voidp joystick)

module Hat = struct
  type t = int
  let centered = sdl_hat_centered
  let up = sdl_hat_up
  let right = sdl_hat_right
  let down = sdl_hat_down
  let left = sdl_hat_left
  let rightup = sdl_hat_rightup
  let rightdown = sdl_hat_rightdown
  let leftup = sdl_hat_leftup
  let leftdown = sdl_hat_leftdown
end

let joystick_close =
  foreign "SDL_JoystickClose" (joystick @-> returning void)

let joystick_event_state =
  foreign "SDL_JoystickEventState" (int @-> returning nat_to_ok)

let joystick_get_event_state () =
  joystick_event_state sdl_query

let joystick_set_event_state s =
  joystick_event_state s

let joystick_get_attached =
  foreign "SDL_JoystickGetAttached" (joystick @-> returning bool)

let joystick_get_axis =
  foreign "SDL_JoystickGetAxis" (joystick @-> int @-> returning int16_t)

let joystick_get_ball =
  foreign "SDL_JoystickGetBall"
    (joystick @-> int @-> (ptr int) @-> (ptr int) @-> returning int)

let joystick_get_ball j i =
  let x = allocate int 0 in
  let y = allocate int 0 in
  match joystick_get_ball j i x y with
  | 0 -> Ok (!@ x, !@ y) | _ -> error ()

let joystick_get_button =
  foreign "SDL_JoystickGetButton"
    (joystick @-> int @-> returning int_as_uint8_t)

let joystick_get_device_guid =
  foreign "SDL_JoystickGetDeviceGUID" (int @-> returning joystick_guid)

let joystick_get_guid =
  foreign "SDL_JoystickGetGUID" (joystick @-> returning joystick_guid)

let joystick_get_guid_from_string =
  foreign "SDL_JoystickGetGUIDFromString" (string @-> returning joystick_guid)

let joystick_get_guid_string =
  foreign "SDL_JoystickGetGUIDString"
    (joystick_guid @-> ptr char @-> int @-> returning void)

let joystick_get_guid_string guid =
  let len = 33 in
  let s = CArray.start (CArray.make char 33) in
  joystick_get_guid_string guid s len;
  coerce (ptr char) string s

let joystick_get_hat =
  foreign "SDL_JoystickGetHat" (joystick @-> int @-> returning int_as_uint8_t)

let joystick_instance_id =
  foreign "SDL_JoystickInstanceID" (joystick @-> returning joystick_id)

let joystick_instance_id j =
  match joystick_instance_id j with
  | n when n < 0l -> error () | n -> Ok n

let joystick_name =
  foreign "SDL_JoystickName" (joystick @-> returning (some_to_ok string_opt))

let joystick_name_for_index =
  foreign "SDL_JoystickNameForIndex" (int @-> returning (some_to_ok string_opt))

let joystick_num_axes =
  foreign "SDL_JoystickNumAxes" (joystick @-> returning nat_to_ok)

let joystick_num_balls =
  foreign "SDL_JoystickNumBalls" (joystick @-> returning nat_to_ok)

let joystick_num_buttons =
  foreign "SDL_JoystickNumButtons" (joystick @-> returning nat_to_ok)

let joystick_num_hats =
  foreign "SDL_JoystickNumHats" (joystick @-> returning nat_to_ok)

let joystick_open =
  foreign "SDL_JoystickOpen" (int @-> returning (some_to_ok joystick_opt))

let joystick_update =
  foreign "SDL_JoystickUpdate" (void @-> returning void)

let num_joysticks =
  foreign "SDL_NumJoysticks" (void @-> returning nat_to_ok)

(* Game controller *)

type game_controller = unit ptr
let game_controller : game_controller typ = ptr void
let game_controller_opt : game_controller option typ = ptr_opt void

let unsafe_game_controller_of_ptr addr : game_controller =
  ptr_of_raw_address addr
let unsafe_ptr_of_game_controller game_controller =
  raw_address_of_ptr (to_voidp game_controller)

type _button_bind
let button_bind : _button_bind structure typ =
  structure "SDL_GameControllerBindType"
let button_bind_bind_type = field button_bind "bindType" int
let button_bind_value1 = field button_bind "value1" int  (* simplified enum *)
let button_bind_value2 = field button_bind "value2" int
let () = seal button_bind

module Controller = struct
  type bind_type = int
  let bind_type_none = sdl_controller_bindtype_none
  let bind_type_button = sdl_controller_bindtype_button
  let bind_type_axis = sdl_controller_bindtype_axis
  let bind_type_hat = sdl_controller_bindtype_hat

  type axis = int
  let axis_invalid = sdl_controller_axis_invalid
  let axis_left_x = sdl_controller_axis_leftx
  let axis_left_y = sdl_controller_axis_lefty
  let axis_right_x = sdl_controller_axis_rightx
  let axis_right_y = sdl_controller_axis_righty
  let axis_trigger_left = sdl_controller_axis_triggerleft
  let axis_trigger_right = sdl_controller_axis_triggerright
  let axis_max = sdl_controller_axis_max

  type button = int
  let button_invalid = sdl_controller_button_invalid
  let button_a = sdl_controller_button_a
  let button_b = sdl_controller_button_b
  let button_x = sdl_controller_button_x
  let button_y = sdl_controller_button_y
  let button_back = sdl_controller_button_back
  let button_guide = sdl_controller_button_guide
  let button_start = sdl_controller_button_start
  let button_left_stick = sdl_controller_button_leftstick
  let button_right_stick = sdl_controller_button_rightstick
  let button_left_shoulder = sdl_controller_button_leftshoulder
  let button_right_shoulder = sdl_controller_button_rightshoulder
  let button_dpad_up = sdl_controller_button_dpad_up
  let button_dpad_down = sdl_controller_button_dpad_down
  let button_dpad_left = sdl_controller_button_dpad_left
  let button_dpad_right = sdl_controller_button_dpad_right
  let button_max = sdl_controller_button_max

  type button_bind = _button_bind structure
  let bind_type v = getf v button_bind_bind_type
  let bind_button_value v = getf v button_bind_value1
  let bind_axis_value v = getf v button_bind_value1
  let bind_hat_value v = getf v button_bind_value1, getf v button_bind_value2
end

let game_controller_add_mapping =
  foreign "SDL_GameControllerAddMapping" (string @-> returning bool_to_ok)

let game_controller_add_mapping_from_file =
  foreign "SDL_GameControllerAddMappingsFromFile"
    ~stub (string @-> returning nat_to_ok)

let game_controller_add_mapping_from_rw =
  foreign "SDL_GameControllerAddMappingsFromRW"
    ~stub (rw_ops @-> bool @-> returning nat_to_ok)

let game_controller_close =
  foreign "SDL_GameControllerClose" (game_controller @-> returning void)

let game_controller_event_state =
  foreign "SDL_GameControllerEventState" (int @-> returning nat_to_ok)

let game_controller_get_event_state () =
  game_controller_event_state sdl_query

let game_controller_set_event_state t =
  game_controller_event_state t

let game_controller_get_attached =
  foreign "SDL_GameControllerGetAttached" (game_controller @-> returning bool)

let game_controller_get_axis =
  foreign "SDL_GameControllerGetAxis"
    (game_controller @-> int @-> returning int16_t)

let game_controller_get_axis_from_string =
  foreign "SDL_GameControllerGetAxisFromString"
    (string @-> returning int)

let game_controller_get_bind_for_axis =
  foreign "SDL_GameControllerGetBindForAxis"
    (game_controller @-> int @-> returning button_bind)

let game_controller_get_bind_for_button =
  foreign "SDL_GameControllerGetBindForButton"
    (game_controller @-> int @-> returning button_bind)

let game_controller_get_button =
  foreign "SDL_GameControllerGetButton"
    (game_controller @-> int @-> returning int_as_uint8_t)

let game_controller_get_button_from_string =
  foreign "SDL_GameControllerGetButtonFromString" (string @-> returning int)

let game_controller_get_joystick =
  foreign "SDL_GameControllerGetJoystick"
    (game_controller @-> returning (some_to_ok joystick_opt))

let game_controller_get_string_for_axis =
  foreign "SDL_GameControllerGetStringForAxis" (int @-> returning string_opt)

let game_controller_get_string_for_button =
  foreign "SDL_GameControllerGetStringForButton" (int @-> returning string_opt)

let game_controller_mapping =
  foreign "SDL_GameControllerMapping"
    (game_controller @-> returning (some_to_ok string_opt))

let game_controller_mapping_for_guid =
  foreign "SDL_GameControllerMappingForGUID"
    (joystick_guid @-> returning (some_to_ok string_opt))

let game_controller_name =
  foreign "SDL_GameControllerName"
    (game_controller @-> returning (some_to_ok string_opt))

let game_controller_name_for_index =
  foreign "SDL_GameControllerNameForIndex"
    (int @-> returning (some_to_ok string_opt))

let game_controller_open =
  foreign "SDL_GameControllerOpen"
    (int @-> returning (some_to_ok game_controller_opt))

let game_controller_update =
  foreign "SDL_GameControllerUpdate" (void @-> returning void)

let is_game_controller =
  foreign "SDL_IsGameController" (int @-> returning bool)

(* Events *)

type event_type = int
let event_type : event_type typ = int_as_uint32_t

module Event = struct

  (* Event structures *)

  module Common = struct
    type t
    let t : t structure typ = structure "SDL_CommonEvent"
    let typ = field t "type" int_as_uint32_t
    let timestamp = field t "timestamp" int32_as_uint32_t
    let () = seal t
  end

  module Controller_axis_event = struct
    type t
    let t : t structure typ = structure "SDL_ControllerAxisEvent"
    let _ = field t "type" int_as_uint32_t
    let _ = field t "timestamp" int32_as_uint32_t
    let which = field t "which" joystick_id
    let axis = field t "axis" int_as_uint8_t
    let _ = field t "padding1" uint8_t
    let _ = field t "padding2" uint8_t
    let _ = field t "padding3" uint8_t
    let value = field t "value" int16_t
    let _ = field t "padding4" uint16_t
    let () = seal t
  end

  module Controller_button_event = struct
    type t
    let t : t structure typ = structure "SDL_ControllerButtonEvent"
    let _ = field t "type" int_as_uint32_t
    let _ = field t "timestamp" int32_as_uint32_t
    let which = field t "which" joystick_id
    let button = field t "button" int_as_uint8_t
    let state = field t "state" int_as_uint8_t
    let _ = field t "padding1" uint8_t
    let _ = field t "padding2" uint8_t
    let () = seal t
  end

  module Controller_device_event = struct
    type t
    let t : t structure typ = structure "SDL_ControllerDeviceEvent"
    let _ = field t "type" int_as_uint32_t
    let _ = field t "timestamp" int32_as_uint32_t
    let which = field t "which" joystick_id
    let () = seal t
  end

  module Dollar_gesture_event = struct
    type t
    let t : t structure typ = structure "SDL_DollarGestureEvent"
    let _ = field t "type" int_as_uint32_t
    let _ = field t "timestamp" int32_as_uint32_t
    let touch_id = field t "touchId" touch_id
    let gesture_id = field t "gestureId" gesture_id
    let num_fingers = field t "numFingers" int_as_uint32_t
    let error = field t "error" float
    let x = field t "x" float
    let y = field t "y" float
    let () = seal t
  end

  module Drop_event = struct
    type t
    let t : t structure typ = structure "SDL_DropEvent"
    let _ = field t "type" int_as_uint32_t
    let _ = field t "timestamp" int32_as_uint32_t
    let file = field t "file" (ptr char)
    let () = seal t
  end

  module Keyboard_event = struct
    type t
    let t : t structure typ = structure "SDL_KeyboardEvent"
    let _ = field t "type" int_as_uint32_t
    let _ = field t "timestamp" int32_as_uint32_t
    let window_id = field t "windowID" int_as_uint32_t
    let state = field t "state" int_as_uint8_t
    let repeat = field t "repeat" int_as_uint8_t
    let padding2 = field t "padding2" uint8_t
    let padding3 = field t "padding3" uint8_t
    (* We inline the definition of SDL_Keysym *)
    let scancode = field t "scancode" scancode
    let keycode = field t "sym" keycode
    let keymod = field t "mod" keymod
    let unused = field t "unused" uint32_t
    let () = seal t
  end

  module Joy_axis_event = struct
    type t
    let t : t structure typ = structure "SDL_JoyAxisEvent"
    let _ = field t "type" int_as_uint32_t
    let _ = field t "timestamp" int32_as_uint32_t
    let which = field t "which" joystick_id
    let axis = field t "axis" int_as_uint8_t
    let _ = field t "padding1" uint8_t
    let _ = field t "padding2" uint8_t
    let _ = field t "padding3" uint8_t
    let value = field t "value" int16_t
    let _ = field t "padding4" uint16_t
    let () = seal t
  end

  module Joy_ball_event = struct
    type t
    let t : t structure typ = structure "SDL_JoyBallEvent"
    let _ = field t "type" int_as_uint32_t
    let _ = field t "timestamp" int32_as_uint32_t
    let which = field t "which" joystick_id
    let ball = field t "ball" int_as_uint8_t
    let _ = field t "padding1" uint8_t
    let _ = field t "padding2" uint8_t
    let _ = field t "padding3" uint8_t
    let xrel = field t "xrel" int16_t
    let yrel = field t "yrel" int16_t
    let () = seal t
  end

  module Joy_button_event = struct
    type t
    let t : t structure typ = structure "SDL_JoyButtonEvent"
    let _ = field t "type" int_as_uint32_t
    let _ = field t "timestamp" int32_as_uint32_t
    let which = field t "which" joystick_id
    let button = field t "button" int_as_uint8_t
    let state = field t "state" int_as_uint8_t
    let _ = field t "padding1" uint8_t
    let _ = field t "padding2" uint8_t
    let () = seal t
  end

  module Joy_device_event = struct
    type t
    let t : t structure typ = structure "SDL_JoyDeviceEvent"
    let _ = field t "type" int_as_uint32_t
    let _ = field t "timestamp" int32_as_uint32_t
    let which = field t "which" joystick_id
    let () = seal t
  end

  module Joy_hat_event = struct
    type t
    let t : t structure typ = structure "SDL_JoyHatEvent"
    let _ = field t "type" int_as_uint32_t
    let _ = field t "timestamp" int32_as_uint32_t
    let which = field t "which" joystick_id
    let hat = field t "hat" int_as_uint8_t
    let value = field t "value" int_as_uint8_t
    let _ = field t "padding1" uint8_t
    let _ = field t "padding2" uint8_t
    let () = seal t
  end

  module Mouse_button_event = struct
    type t
    let t : t structure typ = structure "SDL_MouseButtonEvent"
    let _ = field t "type" int_as_uint32_t
    let _ = field t "timestamp" int32_as_uint32_t
    let window_id = field t "windowID" int_as_uint32_t
    let which = field t "which" int32_as_uint32_t
    let button = field t "button" int_as_uint8_t
    let state = field t "state" int_as_uint8_t
    let clicks = field t "clicks" int_as_uint8_t
    let _ = field t "padding1" int_as_uint8_t
    let x = field t "x" int_as_int32_t
    let y = field t "y" int_as_int32_t
    let () = seal t
  end

  module Mouse_motion_event = struct
    type t
    let t : t structure typ = structure "SDL_MouseMotionEvent"
    let _ = field t "type" int_as_uint32_t
    let _ = field t "timestamp" int32_as_uint32_t
    let window_id = field t "windowID" int_as_uint32_t
    let which = field t "which" int32_as_uint32_t
    let state = field t "state" int32_as_uint32_t
    let x = field t "x" int_as_int32_t
    let y = field t "y" int_as_int32_t
    let xrel = field t "xrel" int_as_int32_t
    let yrel = field t "yrel" int_as_int32_t
    let () = seal t
  end

  module Mouse_wheel_event = struct
    type t
    let t : t structure typ = structure "SDL_MouseWheelEvent"
    let _ = field t "type" int_as_uint32_t
    let _ = field t "timestamp" int32_as_uint32_t
    let window_id = field t "windowID" int_as_uint32_t
    let which = field t "which" int32_as_uint32_t
    let x = field t "x" int_as_int32_t
    let y = field t "y" int_as_int32_t
    let () = seal t
  end

  module Multi_gesture_event = struct
    type t
    let t : t structure typ = structure "SDL_MultiGestureEvent"
    let _ = field t "type" int_as_uint32_t
    let _ = field t "timestamp" int32_as_uint32_t
    let touch_id = field t "touchId" touch_id
    let dtheta = field t "dTheta" float
    let ddist = field t "ddist" float
    let x = field t "x" float
    let y = field t "y" float
    let num_fingers = field t "numFingers" int_as_uint16_t
    let _ = field t "padding" uint16_t
    let () = seal t
  end

  module Quit_event = struct
    type t
    let t : t structure typ = structure "SDL_QuitEvent"
    let _ = field t "type" int_as_uint32_t
    let _ = field t "timestamp" int32_as_uint32_t
    let () = seal t
  end

  module Sys_wm_event = struct
    type t
    let t : t structure typ = structure "SDL_SysWMEvent"
    let _ = field t "type" int_as_uint32_t
    let _ = field t "timestamp" int32_as_uint32_t
    let _ = field t "msg" (ptr void)
    let () = seal t
  end

  module Text_editing_event = struct
    type t
    let t : t structure typ = structure "SDL_TextEditingEvent"
    let _ = field t "type" int_as_uint32_t
    let _ = field t "timestamp" int32_as_uint32_t
    let window_id = field t "windowID" int_as_uint32_t
    let text = field t "text" (string_as_char_array
                                 sdl_texteditingevent_text_size)
    let start = field t "start" int_as_int32_t
    let length = field t "end" int_as_int32_t
    let () = seal t
  end

  module Text_input_event = struct
    type t
    let t : t structure typ = structure "SDL_TextIfmtsnputEvent"
    let _ = field t "type" int_as_uint32_t
    let _ = field t "timestamp" int32_as_uint32_t
    let window_id = field t "windowID" int_as_uint32_t
    let text = field t "text" (string_as_char_array
                                 sdl_textinputevent_text_size)
    let () = seal t
  end

  module Touch_finger_event = struct
    type t
    let t : t structure typ = structure "SDL_TouchFingerEvent"
    let _ = field t "type" int_as_uint32_t
    let _ = field t "timestamp" int32_as_uint32_t
    let touch_id = field t "touchId" touch_id
    let finger_id = field t "fingerId" finger_id
    let x = field t "x" float
    let y = field t "y" float
    let dx = field t "dx" float
    let dy = field t "dy" float
    let pressure = field t "pressure" float
    let () = seal t
  end

  module User_event = struct
    type t
    let t : t structure typ = structure "SDL_UserEvent"
    let _ = field t "type" int_as_uint32_t
    let _ = field t "timestamp" int32_as_uint32_t
    let window_id = field t "windowID" int_as_uint32_t
    let code = field t "code" int_as_int32_t
    let _ = field t "data1" (ptr void)
    let _ = field t "data2" (ptr void)
    let () = seal t
  end

  module Window_event = struct
    type t
    let t : t structure typ = structure "SDL_WindowEvent"
    let _ = field t "type" int_as_uint32_t
    let _ = field t "timestamp" int32_as_uint32_t
    let window_id = field t "windowID" int_as_uint32_t
    let event = field t "event" int_as_uint8_t
    let padding1 = field t "padding1" uint8_t
    let padding2 = field t "padding2" uint8_t
    let padding3 = field t "padding3" uint8_t
    let data1 = field t "data1" int32_t
    let data2 = field t "data2" int32_t
    let () = seal t
  end

  type t
  let t : t union typ = union "SDL_Event"
  let typ = field t "type" int_as_uint32_t
  let common = field t "common" Common.t
  let controller_axis_event = field t "caxis" Controller_axis_event.t
  let controller_button_event = field t "cbutton" Controller_button_event.t
  let controller_device_event = field t "cdevice" Controller_device_event.t
  let dollar_gesture_event = field t "dgesture" Dollar_gesture_event.t
  let drop_event = field t "drop" Drop_event.t
  let joy_axis_event = field t "jaxis" Joy_axis_event.t
  let joy_ball_event = field t "jball" Joy_ball_event.t
  let joy_button_event = field t "jbutton" Joy_button_event.t
  let joy_device_event = field t "jdevice" Joy_device_event.t
  let joy_hat_event = field t "jhat" Joy_hat_event.t
  let keyboard_event = field t "key" Keyboard_event.t
  let mouse_button_event = field t "button" Mouse_button_event.t
  let mouse_motion_event = field t "motion" Mouse_motion_event.t
  let mouse_wheel_event = field t "wheel" Mouse_wheel_event.t
  let multi_gesture_event = field t "mgesture" Multi_gesture_event.t
  let quit_event = field t "quit" Quit_event.t
  let sys_wm_event = field t "syswm" Sys_wm_event.t
  let text_editing_event = field t "edit" Text_editing_event.t
  let text_input_event = field t "text" Text_input_event.t
  let touch_finger_event = field t "tfinger" Touch_finger_event.t
  let user_event = field t "user" User_event.t
  let window_event = field t "window" Window_event.t
  let padding = field t "padding" (abstract "padding" tsdl_sdl_event_size 1)
  let () = seal t

  let create () = make t
  let opt_addr = function
  | None -> coerce (ptr void) (ptr t) null
  | Some v -> addr v

  type _ field =
      F : (* existential to hide the 'a structure *)
        (('a structure, t union) Ctypes.field *
         ('b, 'a structure) Ctypes.field) -> 'b field

  let get e (F (s, f)) = getf (getf e s) f
  let set e (F (s, f)) v = setf (getf e s) f v

  (* Aliases *)

  let first_event = sdl_firstevent
  let last_event = sdl_lastevent

  (* Common *)

  let typ  = F (common, Common.typ)
  let timestamp = F (common, Common.timestamp)

  (* Application events. *)

  let app_terminating = sdl_app_terminating
  let app_low_memory = sdl_app_lowmemory
  let app_will_enter_background = sdl_app_willenterbackground
  let app_did_enter_background = sdl_app_didenterbackground
  let app_will_enter_foreground = sdl_app_willenterforeground
  let app_did_enter_foreground = sdl_app_didenterforeground

  (* Clipboard events *)

  let clipboard_update = sdl_clipboardupdate

  (* Controller events *)

  let controller_axis_motion = sdl_controlleraxismotion
  let controller_button_down = sdl_controllerbuttondown
  let controller_button_up = sdl_controllerbuttonup
  let controller_device_added = sdl_controllerdeviceadded
  let controller_device_remapped = sdl_controllerdeviceremapped
  let controller_device_removed = sdl_controllerdeviceremoved

  let controller_axis_which =
    F (controller_axis_event, Controller_axis_event.which)
  let controller_axis_axis =
    F (controller_axis_event, Controller_axis_event.axis)
  let controller_axis_value =
    F (controller_axis_event, Controller_axis_event.value)

  let controller_button_which =
    F (controller_button_event, Controller_button_event.which)
  let controller_button_button =
    F (controller_button_event, Controller_button_event.button)
  let controller_button_state =
    F (controller_button_event, Controller_button_event.state)

  let controller_device_which =
    F (controller_device_event, Controller_device_event.which)

  (* Dollar gesture events *)

  let dollar_gesture = sdl_dollargesture
  let dollar_record = sdl_dollarrecord

  let dollar_gesture_touch_id =
    F (dollar_gesture_event, Dollar_gesture_event.touch_id)
  let dollar_gesture_gesture_id =
    F (dollar_gesture_event, Dollar_gesture_event.gesture_id)
  let dollar_gesture_num_fingers =
    F (dollar_gesture_event, Dollar_gesture_event.num_fingers)
  let dollar_gesture_error =
    F (dollar_gesture_event, Dollar_gesture_event.error)
  let dollar_gesture_x = F (dollar_gesture_event, Dollar_gesture_event.x)
  let dollar_gesture_y = F (dollar_gesture_event, Dollar_gesture_event.y)

  (* Drop file event *)

  let drop_file = sdl_dropfile
  let drop_file_file = F (drop_event, Drop_event.file)

  let drop_file_free e =
    sdl_free (to_voidp (get e drop_file_file))

  let drop_file_file e =
    let sp = get e drop_file_file in
    if ptr_compare (to_voidp sp) null = 0 then invalid_arg err_drop_file else
    coerce (ptr char) string (get e drop_file_file)

  (* Touch events *)

  let finger_down = sdl_fingerdown
  let finger_motion = sdl_fingermotion
  let finger_up = sdl_fingerup

  let touch_finger_touch_id = F (touch_finger_event,Touch_finger_event.touch_id)
  let touch_finger_finger_id =
    F (touch_finger_event, Touch_finger_event.finger_id)
  let touch_finger_x = F (touch_finger_event, Touch_finger_event.x)
  let touch_finger_y = F (touch_finger_event, Touch_finger_event.y)
  let touch_finger_dx = F (touch_finger_event, Touch_finger_event.dx)
  let touch_finger_dy = F (touch_finger_event, Touch_finger_event.dy)
  let touch_finger_pressure =
    F (touch_finger_event, Touch_finger_event.pressure)

  (* Joystick events. *)

  let joy_axis_motion = sdl_joyaxismotion
  let joy_ball_motion = sdl_joyballmotion
  let joy_button_down = sdl_joybuttondown
  let joy_button_up = sdl_joybuttonup
  let joy_device_added = sdl_joydeviceadded
  let joy_device_removed = sdl_joydeviceremoved
  let joy_hat_motion = sdl_joyhatmotion

  let joy_axis_which = F (joy_axis_event, Joy_axis_event.which)
  let joy_axis_axis = F (joy_axis_event, Joy_axis_event.axis)
  let joy_axis_value = F (joy_axis_event, Joy_axis_event.value)

  let joy_ball_which = F (joy_ball_event, Joy_ball_event.which)
  let joy_ball_ball = F (joy_ball_event, Joy_ball_event.ball)
  let joy_ball_xrel = F (joy_ball_event, Joy_ball_event.xrel)
  let joy_ball_yrel = F (joy_ball_event, Joy_ball_event.yrel)

  let joy_button_which = F (joy_button_event, Joy_button_event.which)
  let joy_button_button = F (joy_button_event, Joy_button_event.button)
  let joy_button_state = F (joy_button_event, Joy_button_event.state)

  let joy_device_which = F (joy_device_event, Joy_device_event.which)

  let joy_hat_which = F (joy_hat_event, Joy_hat_event.which)
  let joy_hat_hat = F (joy_hat_event, Joy_hat_event.hat)
  let joy_hat_value = F (joy_hat_event, Joy_hat_event.value)

  (* Keyboard events *)

  let key_down = sdl_keydown
  let key_up = sdl_keyup

  let keyboard_window_id = F (keyboard_event, Keyboard_event.window_id)
  let keyboard_repeat = F (keyboard_event, Keyboard_event.repeat)
  let keyboard_state = F (keyboard_event, Keyboard_event.state)
  let keyboard_scancode = F (keyboard_event, Keyboard_event.scancode)
  let keyboard_keycode = F (keyboard_event, Keyboard_event.keycode)
  let keyboard_keymod = F (keyboard_event, Keyboard_event.keymod)

  (* Mouse events *)

  let mouse_button_down = sdl_mousebuttondown
  let mouse_button_up = sdl_mousebuttonup
  let mouse_motion = sdl_mousemotion
  let mouse_wheel = sdl_mousewheel

  let mouse_button_window_id =
    F (mouse_button_event, Mouse_button_event.window_id)
  let mouse_button_which = F (mouse_button_event, Mouse_button_event.which)
  let mouse_button_state = F (mouse_button_event, Mouse_button_event.state)
  let mouse_button_button = F (mouse_button_event, Mouse_button_event.button)
  let mouse_button_clicks = F (mouse_button_event, Mouse_button_event.clicks)
  let mouse_button_x = F (mouse_button_event, Mouse_button_event.x)
  let mouse_button_y = F (mouse_button_event, Mouse_button_event.y)

  let mouse_motion_window_id =
    F (mouse_motion_event, Mouse_motion_event.window_id)
  let mouse_motion_which = F (mouse_motion_event, Mouse_motion_event.which)
  let mouse_motion_state = F (mouse_motion_event, Mouse_motion_event.state)
  let mouse_motion_x = F (mouse_motion_event, Mouse_motion_event.x)
  let mouse_motion_y = F (mouse_motion_event, Mouse_motion_event.y)
  let mouse_motion_xrel = F (mouse_motion_event, Mouse_motion_event.xrel)
  let mouse_motion_yrel = F (mouse_motion_event, Mouse_motion_event.yrel)

  let mouse_wheel_window_id = F (mouse_wheel_event, Mouse_wheel_event.window_id)
  let mouse_wheel_which = F (mouse_wheel_event, Mouse_wheel_event.which)
  let mouse_wheel_x = F (mouse_wheel_event, Mouse_wheel_event.x)
  let mouse_wheel_y = F (mouse_wheel_event, Mouse_wheel_event.y)

  (* Multi gesture events *)

  let multi_gesture = sdl_multigesture

  let multi_gesture_touch_id =
    F (multi_gesture_event, Multi_gesture_event.touch_id)
  let multi_gesture_dtheta = F (multi_gesture_event, Multi_gesture_event.dtheta)
  let multi_gesture_ddist = F (multi_gesture_event, Multi_gesture_event.ddist)
  let multi_gesture_x = F (multi_gesture_event, Multi_gesture_event.x)
  let multi_gesture_y = F (multi_gesture_event, Multi_gesture_event.y)
  let multi_gesture_num_fingers =
    F (multi_gesture_event, Multi_gesture_event.num_fingers)

  (* Quit events *)

  let quit = sdl_quit

  (* System window manager events *)

  let sys_wm_event = sdl_syswmevent

  (* Text events *)

  let text_editing = sdl_textediting
  let text_input = sdl_textinput

  let text_editing_window_id =
    F (text_editing_event, Text_editing_event.window_id)
  let text_editing_text = F (text_editing_event, Text_editing_event.text)
  let text_editing_start = F (text_editing_event, Text_editing_event.start)
  let text_editing_length = F (text_editing_event, Text_editing_event.length)

  let text_input_window_id = F (text_input_event, Text_input_event.window_id)
  let text_input_text = F (text_input_event, Text_input_event.text)

  (* User events *)

  let user_window_id = F (user_event, User_event.window_id)
  let user_code = F (user_event, User_event.code)
  let user_event = sdl_userevent

  (* Window events *)

  type window_event_id = int
  let window_event_shown = sdl_windowevent_shown
  let window_event_hidden = sdl_windowevent_hidden
  let window_event_exposed = sdl_windowevent_exposed
  let window_event_moved = sdl_windowevent_moved
  let window_event_resized = sdl_windowevent_resized
  let window_event_size_changed = sdl_windowevent_size_changed
  let window_event_minimized = sdl_windowevent_minimized
  let window_event_maximized = sdl_windowevent_maximized
  let window_event_restored = sdl_windowevent_restored
  let window_event_enter = sdl_windowevent_enter
  let window_event_leave = sdl_windowevent_leave
  let window_event_focus_gained = sdl_windowevent_focus_gained
  let window_event_focus_lost = sdl_windowevent_focus_lost
  let window_event_close = sdl_windowevent_close

  let window_window_id = F (window_event, Window_event.window_id)
  let window_event_id = F (window_event, Window_event.event)
  let window_data1 = F (window_event, Window_event.data1)
  let window_data2 = F (window_event, Window_event.data2)

  let window_event = sdl_windowevent

  (* Window event id enum *)

  let enum_of_window_event_id =
    let add acc (k, v) = Imap.add k v acc in
    let enums = [
      window_event_shown, `Shown;
      window_event_hidden, `Hidden;
      window_event_exposed, `Exposed;
      window_event_moved, `Moved;
      window_event_resized, `Resized;
      window_event_size_changed, `Size_changed;
      window_event_minimized, `Minimized;
      window_event_maximized, `Maximized;
      window_event_restored, `Restored;
      window_event_enter, `Enter;
      window_event_leave, `Leave;
      window_event_focus_gained, `Focus_gained;
      window_event_focus_lost, `Focus_lost;
      window_event_close, `Close; ]
    in
    List.fold_left add Imap.empty enums

  let window_event_enum id =
    try Imap.find id enum_of_window_event_id with Not_found -> assert false

  (* Event type enum *)

  let enum_of_event_type =
    let add acc (k, v) = Imap.add k v acc in
    let enums = [ app_terminating, `App_terminating;
                  app_low_memory, `App_low_memory;
                  app_will_enter_background, `App_will_enter_background;
                  app_did_enter_background, `App_did_enter_background;
                  app_will_enter_foreground, `App_will_enter_foreground;
                  app_did_enter_foreground, `App_did_enter_foreground;
                  clipboard_update, `Clipboard_update;
                  controller_axis_motion, `Controller_axis_motion;
                  controller_button_down, `Controller_button_down;
                  controller_button_up, `Controller_button_up;
                  controller_device_added, `Controller_device_added;
                  controller_device_remapped, `Controller_device_remapped;
                  controller_device_removed, `Controller_device_removed;
                  dollar_gesture, `Dollar_gesture;
                  dollar_record, `Dollar_record;
                  drop_file, `Drop_file;
                  finger_down, `Finger_down;
                  finger_motion, `Finger_motion;
                  finger_up, `Finger_up;
                  joy_axis_motion, `Joy_axis_motion;
                  joy_ball_motion, `Joy_ball_motion;
                  joy_button_down, `Joy_button_down;
                  joy_button_up, `Joy_button_up;
                  joy_device_added, `Joy_device_added;
                  joy_device_removed, `Joy_device_removed;
                  joy_hat_motion, `Joy_hat_motion;
                  key_down, `Key_down;
                  key_up, `Key_up;
                  mouse_button_down, `Mouse_button_down;
                  mouse_button_up, `Mouse_button_up;
                  mouse_motion, `Mouse_motion;
                  mouse_wheel, `Mouse_wheel;
                  multi_gesture, `Multi_gesture;
                  sys_wm_event, `Sys_wm_event;
                  text_editing, `Text_editing;
                  text_input, `Text_input;
                  user_event, `User_event;
                  quit, `Quit;
                  window_event, `Window_event; ]
    in
    List.fold_left add Imap.empty enums

  let enum t = try Imap.find t enum_of_event_type with Not_found -> `Unknown

end

type event = Event.t union

let event_state =
  foreign "SDL_EventState" (event_type @-> int @-> returning int_as_uint8_t)

let get_event_state e =
  event_state e sdl_query

let set_event_state e s =
  ignore (event_state e s)

let flush_event =
  foreign "SDL_FlushEvent" (event_type @-> returning void)

let flush_events =
  foreign "SDL_FlushEvents" (event_type @-> event_type @-> returning void)

let has_event =
  foreign "SDL_HasEvent" (event_type @-> returning bool)

let has_events =
  foreign "SDL_HasEvents" (event_type @-> event_type @-> returning bool)

let poll_event =
  foreign "SDL_PollEvent" (ptr Event.t @-> returning bool)

let poll_event e =
  poll_event (Event.opt_addr e)

let pump_events =
  foreign "SDL_PumpEvents" (void @-> returning void)

let push_event =
  foreign "SDL_PushEvent" (ptr Event.t @-> returning bool_to_ok)

let push_event e =
  push_event (addr e)

let register_events =
  foreign "SDL_RegisterEvents" (int @-> returning uint32_t)

let register_event () = match Unsigned.UInt32.to_int32 (register_events 1) with
| -1l -> None | t -> Some (Int32.to_int t)

let wait_event =
  foreign ~release_runtime_lock:true
    "SDL_WaitEvent" (ptr Event.t @-> returning int)

let wait_event e = match wait_event (Event.opt_addr e) with
| 1 -> Ok () | _ -> error ()

let wait_event_timeout =
  foreign "SDL_WaitEventTimeout" ~release_runtime_lock:true
    (ptr Event.t @-> int @-> returning bool)

let wait_event_timeout e t =
  wait_event_timeout (Event.opt_addr e) t

(* Force feedback *)

type haptic = unit ptr
let haptic : haptic typ = ptr void
let haptic_opt : haptic option typ = ptr_opt void

let unsafe_haptic_of_ptr addr : haptic =
  ptr_of_raw_address addr
let unsafe_ptr_of_haptic haptic =
  raw_address_of_ptr (to_voidp haptic)

module Haptic = struct
  let infinity = -1l

  (* Features *)

  type feature = int
  let gain = sdl_haptic_gain
  let autocenter = sdl_haptic_autocenter
  let status = sdl_haptic_status
  let pause = sdl_haptic_pause

  (* Directions *)

  type direction_type = int
  let polar = sdl_haptic_polar
  let cartesian = sdl_haptic_cartesian
  let spherical = sdl_haptic_spherical

  module Direction = struct
    type _t
    type t = _t structure
    let t : _t structure typ = structure "SDL_HapticDirection"
    let typ = field t "type" int_as_uint8_t
    let dir_0 = field t "dir0" int32_t
    let dir_1 = field t "dir1" int32_t
    let dir_2 = field t "dir2" int32_t
    let () = seal t

    let create typv d0 d1 d2 =
      let d = make t in
      setf d typ typv;
      setf d dir_0 d0;
      setf d dir_1 d1;
      setf d dir_2 d2;
      d

    let typ d = getf d typ
    let dir_0 d = getf d dir_0
    let dir_1 d = getf d dir_1
    let dir_2 d = getf d dir_2
  end

  (* Effects *)

  module Constant = struct
    type t
    let t : t structure typ = structure "SDL_HapticConstant"
    let typ = field t "type" int_as_uint16_t
    let direction = field t "direction" Direction.t
    let length = field t "length" int32_as_uint32_t
    let delay = field t "delay" int_as_uint16_t
    let button = field t "button" int_as_uint16_t
    let interval = field t "interval" int_as_uint16_t

    let level = field t "level" int16_t
    let attack_length = field t "attack_length" int_as_uint16_t
    let attack_level = field t "attack_level" int_as_uint16_t
    let fade_length = field t "fade_length" int_as_uint16_t
    let fade_level = field t "fade_level" int_as_uint16_t
    let () = seal t
  end

  module Periodic = struct
    type t
    let t : t structure typ = structure "SDL_HapticPeriodic"
    let typ = field t "type" int_as_uint16_t
    let direction = field t "direction" Direction.t
    let length = field t "length" int32_as_uint32_t
    let delay = field t "delay" int_as_uint16_t
    let button = field t "button" int_as_uint16_t
    let interval = field t "interval" int_as_uint16_t

    let period = field t "period" int_as_uint16_t
    let magnitude = field t "magnitude" int16_t
    let offset = field t "offset" int16_t
    let phase = field t "phase" int_as_uint16_t
    let attack_length = field t "attack_length" int_as_uint16_t
    let attack_level = field t "attack_level" int_as_uint16_t
    let fade_length = field t "fade_length" int_as_uint16_t
    let fade_level = field t "fade_level" int_as_uint16_t
    let () = seal t
  end

  module Condition = struct
    type t
    let t : t structure typ = structure "SDL_HapticCondition"
    let typ = field t "type" int_as_uint16_t
    let direction = field t "direction" Direction.t
    let length = field t "length" int32_as_uint32_t
    let delay = field t "delay" int_as_uint16_t
    let button = field t "button" int_as_uint16_t
    let interval = field t "interval" int_as_uint16_t

    let right_sat_0 = field t "right_sat[0]" int_as_uint16_t
    let right_sat_1 = field t "right_sat[1]" int_as_uint16_t
    let right_sat_2 = field t "right_sat[2]" int_as_uint16_t
    let left_sat_0 = field t "left_sat[0]" int_as_uint16_t
    let left_sat_1 = field t "left_sat[1]" int_as_uint16_t
    let left_sat_2 = field t "left_sat[2]" int_as_uint16_t
    let right_coeff_0 = field t "right_coeff[0]" int16_t
    let right_coeff_1 = field t "right_coeff[1]" int16_t
    let right_coeff_2 = field t "right_coeff[2]" int16_t
    let left_coeff_0 = field t "left_coeff[0]" int16_t
    let left_coeff_1 = field t "left_coeff[1]" int16_t
    let left_coeff_2 = field t "left_coeff[2]" int16_t
    let deadband_0 = field t "deadband[0]" int_as_uint16_t
    let deadband_1 = field t "deadband[1]" int_as_uint16_t
    let deadband_2 = field t "deadband[2]" int_as_uint16_t
    let center_0 = field t "center[0]" int16_t
    let center_1 = field t "center[1]" int16_t
    let center_2 = field t "center[2]" int16_t
    let () = seal t
  end

  module Ramp = struct
    type t
    let t : t structure typ = structure "SDL_HapticRamp"
    let typ = field t "type" int_as_uint16_t
    let direction = field t "direction" Direction.t
    let length = field t "length" int32_as_uint32_t
    let delay = field t "delay" int_as_uint16_t
    let button = field t "button" int_as_uint16_t
    let interval = field t "interval" int_as_uint16_t

    let start = field t "start" int16_t
    let end_ = field t "end" int16_t
    let attack_length = field t "attack_length" int_as_uint16_t
    let attack_level = field t "attack_level" int_as_uint16_t
    let fade_length = field t "fade_length" int_as_uint16_t
    let fade_level = field t "fade_level" int_as_uint16_t
    let () = seal t
  end

  module Left_right = struct
    type t
    let t : t structure typ = structure "SDL_HapticLeftRight"
    let typ = field t "type" int_as_uint16_t
    let direction = field t "direction" Direction.t
    let length = field t "length" int32_as_uint32_t

    let large_magnitude = field t "large_magnitude" int_as_uint16_t
    let small_magnitude = field t "small_magnitude" int_as_uint16_t
    let () = seal t
  end

  module Custom = struct
    let int_list_as_uint16_t_ptr =
      let read _ = invalid_arg err_read_field in
      let write l =
        let l = List.map Unsigned.UInt16.of_int l in
        let a = CArray.of_list uint16_t l in
        CArray.start a
      in
      view ~read ~write (ptr uint16_t)

    type t
    let t : t structure typ = structure "SDL_HapticCustom"
    let typ = field t "type" int_as_uint16_t
    let direction = field t "direction" Direction.t
    let length = field t "length" int32_as_uint32_t
    let delay = field t "delay" int_as_uint16_t
    let button = field t "button" int_as_uint16_t
    let interval = field t "interval" int_as_uint16_t

    let channels = field t "channels" int_as_uint8_t
    let period = field t "period" int_as_uint16_t
    let samples = field t "samples" int_as_uint16_t
    let data = field t "data" int_list_as_uint16_t_ptr
    let attack_length = field t "attack_length" int_as_uint16_t
    let attack_level = field t "attack_level" int_as_uint16_t
    let fade_length = field t "fade_length" int_as_uint16_t
    let fade_level = field t "fade_level" int_as_uint16_t
    let () = seal t
  end

  module Effect = struct
    type t
    let t : t union typ = union "SDL_HapticEffect"
    let typ = field t "type" int_as_uint16_t
    let constant = field t "constant" Constant.t
    let periodic = field t "periodic" Periodic.t
    let condition = field t "condition" Condition.t
    let ramp = field t "ramp" Ramp.t
    let left_right = field t "condition" Left_right.t
    let custom = field t "custom" Custom.t
    let () = seal t
  end

  type effect_type = int

  let create_effect () = make Effect.t
  let opt_addr = function
  | None -> coerce (ptr void) (ptr Effect.t) null
  | Some v -> addr v

  type _ field =
      F : (* existential to hide the 'a structure *)
        (('a structure, Effect.t union) Ctypes.field *
         ('b, 'a structure) Ctypes.field) -> 'b field

  let get e (F (s, f)) = getf (getf e s) f
  let set e (F (s, f)) v = setf (getf e s) f v
  let typ = F (Effect.constant, Constant.typ) (* same in each enum *)

  (* Constant *)
  let constant = sdl_haptic_constant

  let constant_type = F (Effect.constant, Constant.typ)
  let constant_direction = F (Effect.constant, Constant.direction)
  let constant_length = F (Effect.constant, Constant.length)
  let constant_delay = F (Effect.constant, Constant.delay)
  let constant_button = F (Effect.constant, Constant.button)
  let constant_interval = F (Effect.constant, Constant.interval)
  let constant_level = F (Effect.constant, Constant.level)
  let constant_attack_length = F (Effect.constant, Constant.attack_length)
  let constant_attack_level = F (Effect.constant, Constant.attack_level)
  let constant_fade_length = F (Effect.constant, Constant.fade_length)
  let constant_fade_level = F (Effect.constant, Constant.fade_level)

  (* Periodic *)

  let sine = sdl_haptic_sine
  let left_right = sdl_haptic_leftright
  let triangle = sdl_haptic_triangle
  let sawtooth_up = sdl_haptic_sawtoothup
  let sawtooth_down = sdl_haptic_sawtoothdown

  let periodic_type = F (Effect.periodic, Periodic.typ)
  let periodic_direction = F (Effect.periodic, Periodic.direction)
  let periodic_length = F (Effect.periodic, Periodic.length)
  let periodic_delay = F (Effect.periodic, Periodic.delay)
  let periodic_button = F (Effect.periodic, Periodic.button)
  let periodic_interval = F (Effect.periodic, Periodic.interval)
  let periodic_period = F (Effect.periodic, Periodic.period)
  let periodic_magnitude = F (Effect.periodic, Periodic.magnitude)
  let periodic_offset = F (Effect.periodic, Periodic.offset)
  let periodic_phase = F (Effect.periodic, Periodic.phase)
  let periodic_attack_length = F (Effect.periodic, Periodic.attack_length)
  let periodic_attack_level = F (Effect.periodic, Periodic.attack_level)
  let periodic_fade_length = F (Effect.periodic, Periodic.fade_length)
  let periodic_fade_level = F (Effect.periodic, Periodic.fade_level)

  (* Condition *)

  let spring = sdl_haptic_spring
  let damper = sdl_haptic_damper
  let inertia = sdl_haptic_inertia
  let friction = sdl_haptic_friction

  let condition_type = F (Effect.condition, Condition.typ)
  let condition_direction = F (Effect.condition, Condition.direction)
  let condition_length = F (Effect.condition, Condition.length)
  let condition_delay = F (Effect.condition, Condition.delay)
  let condition_button = F (Effect.condition, Condition.button)
  let condition_interval = F (Effect.condition, Condition.interval)
  let condition_right_sat_0 = F (Effect.condition, Condition.right_sat_0)
  let condition_right_sat_1 = F (Effect.condition, Condition.right_sat_1)
  let condition_right_sat_2 = F (Effect.condition, Condition.right_sat_2)
  let condition_left_sat_0 = F (Effect.condition, Condition.left_sat_0)
  let condition_left_sat_1 = F (Effect.condition, Condition.left_sat_1)
  let condition_left_sat_2 = F (Effect.condition, Condition.left_sat_2)
  let condition_right_coeff_0 = F (Effect.condition, Condition.right_coeff_0)
  let condition_right_coeff_1 = F (Effect.condition, Condition.right_coeff_1)
  let condition_right_coeff_2 = F (Effect.condition, Condition.right_coeff_2)
  let condition_left_coeff_0 = F (Effect.condition, Condition.left_coeff_0)
  let condition_left_coeff_1 = F (Effect.condition, Condition.left_coeff_1)
  let condition_left_coeff_2 = F (Effect.condition, Condition.left_coeff_2)
  let condition_deadband_0 = F (Effect.condition, Condition.deadband_0)
  let condition_deadband_1 = F (Effect.condition, Condition.deadband_1)
  let condition_deadband_2 = F (Effect.condition, Condition.deadband_2)
  let condition_center_0 = F (Effect.condition, Condition.center_0)
  let condition_center_1 = F (Effect.condition, Condition.center_1)
  let condition_center_2 = F (Effect.condition, Condition.center_2)

  (* Ramp *)

  let ramp = sdl_haptic_ramp

  let ramp_type = F (Effect.ramp, Ramp.typ)
  let ramp_direction = F (Effect.ramp, Ramp.direction)
  let ramp_length = F (Effect.ramp, Ramp.length)
  let ramp_delay = F (Effect.ramp, Ramp.delay)
  let ramp_button = F (Effect.ramp, Ramp.button)
  let ramp_interval = F (Effect.ramp, Ramp.interval)
  let ramp_start = F (Effect.ramp, Ramp.start)
  let ramp_end = F (Effect.ramp, Ramp.end_)
  let ramp_attack_length = F (Effect.ramp, Ramp.attack_length)
  let ramp_attack_level = F (Effect.ramp, Ramp.attack_level)
  let ramp_fade_length = F (Effect.ramp, Ramp.fade_length)
  let ramp_fade_level = F (Effect.ramp, Ramp.fade_level)

  (* Left right *)

  let left_right_type = F (Effect.left_right, Left_right.typ)
  let left_right_length = F (Effect.left_right, Left_right.length)
  let left_right_large_magnitude =
    F (Effect.left_right, Left_right.large_magnitude)
  let left_right_small_magnitude =
    F (Effect.left_right, Left_right.small_magnitude)

  (* Custom *)

  let custom = sdl_haptic_custom

  let custom_type = F (Effect.custom, Custom.typ)
  let custom_direction = F (Effect.custom, Custom.direction)
  let custom_length = F (Effect.custom, Custom.length)
  let custom_delay = F (Effect.custom, Custom.delay)
  let custom_button = F (Effect.custom, Custom.button)
  let custom_interval = F (Effect.custom, Custom.interval)
  let custom_channels = F (Effect.custom, Custom.channels)
  let custom_period = F (Effect.custom, Custom.period)
  let custom_samples = F (Effect.custom, Custom.samples)
  let custom_data = F (Effect.custom, Custom.data)
  let custom_attack_length = F (Effect.custom, Custom.attack_length)
  let custom_attack_level = F (Effect.custom, Custom.attack_level)
  let custom_fade_length = F (Effect.custom, Custom.fade_length)
  let custom_fade_level = F (Effect.custom, Custom.fade_level)
end

type haptic_effect = Haptic.Effect.t union

type haptic_effect_id = int
let haptic_effect_id : int typ = int

let haptic_close =
  foreign "SDL_HapticClose" (haptic @-> returning void)

let haptic_destroy_effect =
  foreign "SDL_HapticDestroyEffect"
    (haptic @-> int @-> returning void)

let haptic_effect_supported =
  foreign "SDL_HapticEffectSupported"
    (haptic @-> ptr Haptic.Effect.t @-> returning bool_to_ok)

let haptic_effect_supported h e =
  haptic_effect_supported h (addr e)

let haptic_get_effect_status =
  foreign "SDL_HapticGetEffectStatus"
    (haptic @-> haptic_effect_id @-> returning bool_to_ok)

let haptic_index =
  foreign "SDL_HapticIndex" (haptic @-> returning nat_to_ok)

let haptic_name =
  foreign "SDL_HapticName" (int @-> returning (some_to_ok string_opt))

let haptic_new_effect =
  foreign "SDL_HapticNewEffect"
    (haptic @-> ptr Haptic.Effect.t @-> returning nat_to_ok)

let haptic_new_effect h e =
  haptic_new_effect h (addr e)

let haptic_num_axes =
  foreign "SDL_HapticNumAxes" (haptic @-> returning nat_to_ok)

let haptic_num_effects =
  foreign "SDL_HapticNumEffects" (haptic @-> returning nat_to_ok)

let haptic_num_effects_playing =
  foreign "SDL_HapticNumEffectsPlaying" (haptic @-> returning nat_to_ok)

let haptic_open =
  foreign "SDL_HapticOpen" (int @-> returning (some_to_ok haptic_opt))

let haptic_open_from_joystick =
  foreign "SDL_HapticOpenFromJoystick"
  (joystick @-> returning (some_to_ok haptic_opt))

let haptic_open_from_mouse =
  foreign "SDL_HapticOpenFromMouse"
    (void @-> returning (some_to_ok haptic_opt))

let haptic_opened =
  foreign "SDL_HapticOpened" (int @-> returning int)

let haptic_opened i = match haptic_opened i with
| 0 -> false | 1 -> true | _ -> assert false

let haptic_pause =
  foreign "SDL_HapticPause" (haptic @-> returning zero_to_ok)

let haptic_query =
  foreign "SDL_HapticQuery" (haptic @-> returning int)

let haptic_rumble_init =
  foreign "SDL_HapticRumbleInit" (haptic @-> returning zero_to_ok)

let haptic_rumble_play =
  foreign "SDL_HapticRumblePlay"
    (haptic @-> float @-> int32_t @-> returning zero_to_ok)

let haptic_rumble_stop =
  foreign "SDL_HapticRumbleStop" (haptic @-> returning zero_to_ok)

let haptic_rumble_supported =
  foreign "SDL_HapticRumbleSupported" (haptic @-> returning bool_to_ok)

let haptic_run_effect =
  foreign "SDL_HapticRunEffect"
    (haptic @-> haptic_effect_id  @-> int32_t @-> returning zero_to_ok)

let haptic_set_autocenter =
  foreign "SDL_HapticSetAutocenter" (haptic @-> int @-> returning zero_to_ok)

let haptic_set_gain =
  foreign "SDL_HapticSetGain" (haptic @-> int @-> returning zero_to_ok)

let haptic_stop_all =
  foreign "SDL_HapticStopAll" (haptic @-> returning zero_to_ok)

let haptic_stop_effect =
  foreign "SDL_HapticStopEffect"
    (haptic @-> haptic_effect_id @-> returning zero_to_ok)

let haptic_unpause =
  foreign "SDL_HapticUnpause" (haptic @-> returning zero_to_ok)

let haptic_update_effect =
  foreign "SDL_HapticUpdateEffect"
    (haptic @-> haptic_effect_id @-> ptr Haptic.Effect.t @->
     returning zero_to_ok)

let haptic_update_effect h id e =
  haptic_update_effect h id (addr e)

let joystick_is_haptic =
  foreign "SDL_JoystickIsHaptic"
    (joystick @-> returning bool_to_ok)

let mouse_is_haptic =
  foreign "SDL_MouseIsHaptic" (void @-> returning bool_to_ok)

let num_haptics =
  foreign "SDL_NumHaptics" (void @-> returning nat_to_ok)

(* Audio *)

(* Audio drivers *)

let audio_init =
  foreign "SDL_AudioInit" (string_opt @-> returning zero_to_ok)

let audio_quit =
  foreign "SDL_AudioQuit" (void @-> returning void)

let get_audio_driver =
  foreign "SDL_GetAudioDriver"
    (int @-> returning (some_to_ok string_opt))

let get_current_audio_driver =
  foreign "SDL_GetCurrentAudioDriver" (void @-> returning string_opt)

let get_num_audio_drivers =
  foreign "SDL_GetNumAudioDrivers" (void @-> returning nat_to_ok)

(* Audio devices *)

module Audio = struct
  type status = int
  let stopped = sdl_audio_stopped
  let playing = sdl_audio_playing
  let paused = sdl_audio_paused

  type format = int
  let format = int_as_uint16_t
  let s8 = audio_s8
  let u8 = audio_u8
  let s16_lsb = audio_s16lsb
  let s16_msb = audio_s16msb
  let s16_sys = audio_s16sys
  let s16 = audio_s16
  let s16_lsb = audio_s16lsb
  let u16_lsb = audio_u16lsb
  let u16_msb = audio_u16msb
  let u16_sys = audio_u16sys
  let u16 = audio_u16
  let u16_lsb = audio_u16lsb
  let s32_lsb = audio_s32lsb
  let s32_msb = audio_s32msb
  let s32_sys = audio_s32sys
  let s32 = audio_s32
  let s32_lsb = audio_s32lsb
  let f32_lsb = audio_f32lsb
  let f32_msb = audio_f32msb
  let f32_sys = audio_f32sys
  let f32 = audio_f32

  type allow = int
  let allow = int
  let allow_frequency_change = sdl_audio_allow_frequency_change
  let allow_format_change = sdl_audio_allow_format_change
  let allow_channels_change = sdl_audio_allow_channels_change
  let allow_any_change = sdl_audio_allow_any_change
end

type audio_device_id = Unsigned.uint32
let audio_device_id = uint32_t

type ('a, 'b) audio_spec =
  { as_freq : int;
    as_format : Audio.format;
    as_channels : uint8;
    as_silence : uint8;
    as_samples : uint8;
    as_size : uint32;
    as_ba_kind : ('a, 'b) Bigarray.kind;
    as_callback : (('a, 'b) bigarray -> unit) option; }

let audio_callback =
  (ptr void @-> ptr uint8_t @-> int @-> returning void)

type _audio_spec
let audio_spec : _audio_spec structure typ = structure "SDL_AudioSpec"
let as_freq = field audio_spec "freq" int
let as_format = field audio_spec "format" Audio.format
let as_channels = field audio_spec "channels" int_as_uint8_t
let as_silence = field audio_spec "silence" int_as_uint8_t
let as_samples = field audio_spec "samples" int_as_uint16_t
let _ = field audio_spec "padding" uint16_t
let as_size = field audio_spec "size" int32_as_uint32_t
let as_callback =
  field audio_spec "callback"
    (funptr_opt ~thread_registration:true ~runtime_lock:true audio_callback)

let as_userdata = field audio_spec "userdata" (ptr void)
let () = seal audio_spec

let audio_spec_of_c c as_ba_kind =
  let as_freq = getf c as_freq in
  let as_format = getf c as_format in
  let as_channels = getf c as_channels in
  let as_silence = getf c as_silence in
  let as_samples = getf c as_samples in
  let as_size = getf c as_size in
  let as_callback = None in
  { as_freq; as_format; as_channels; as_silence; as_samples; as_size;
    as_ba_kind; as_callback; }

let audio_spec_to_c a =
  let wrap_cb = match a.as_callback with
  | None -> None
  | Some cb ->
      let kind_bytes = ba_kind_byte_size a.as_ba_kind in
      let ba_ptr_typ = access_ptr_typ_of_ba_kind a.as_ba_kind in
      Some begin fun _ p len ->
        let p = coerce (ptr uint8_t) ba_ptr_typ p in
        let len = len / kind_bytes in
        cb (bigarray_of_ptr array1 len a.as_ba_kind p)
      end
  in
  let c = make audio_spec in
  setf c as_freq a.as_freq;
  setf c as_format a.as_format;
  setf c as_channels a.as_channels;
  setf c as_silence a.as_silence; (* irrelevant *)
  setf c as_samples a.as_samples;
  setf c as_size a.as_size;       (* irrelevant *)
  setf c as_callback wrap_cb;
  setf c as_userdata null;
  c

let close_audio_device =
  foreign "SDL_CloseAudioDevice" (audio_device_id @-> returning void)

let free_wav =
  foreign "SDL_FreeWAV" (ptr void @-> returning void)

let free_wav ba =
  free_wav (to_voidp (bigarray_start array1 ba))

let get_audio_device_name =
  foreign "SDL_GetAudioDeviceName"
    (int @-> bool @-> returning (some_to_ok string_opt))

let get_audio_device_status =
  foreign "SDL_GetAudioDeviceStatus" (audio_device_id @-> returning int)

let get_num_audio_devices =
  foreign "SDL_GetNumAudioDevices" (bool @-> returning nat_to_ok)

let load_wav_rw =
  foreign "SDL_LoadWAV_RW"
    (rw_ops @-> ptr audio_spec @-> ptr (ptr void) @-> ptr uint32_t @->
     returning (some_to_ok (ptr_opt audio_spec)))

let load_wav_rw ops spec =
  let d = allocate (ptr void) null in
  let len = allocate uint32_t Unsigned.UInt32.zero in
  match load_wav_rw ops (addr (audio_spec_to_c spec)) d len with
  | Error _ as e -> e
  | Ok r ->
      let rspec = audio_spec_of_c (!@ r) spec.as_ba_kind in
      let kind_size = ba_kind_byte_size spec.as_ba_kind in
      let len = Unsigned.UInt32.to_int (!@ len) in
      if len mod kind_size <> 0
      then invalid_arg (err_bigarray_data len kind_size)
      else
      let ba_size = len / kind_size in
      let ba_ptr = access_ptr_typ_of_ba_kind spec.as_ba_kind in
      let d = coerce (ptr void)  ba_ptr (!@ d) in
      Ok (rspec, bigarray_of_ptr array1 ba_size spec.as_ba_kind d)

let lock_audio_device =
  foreign "SDL_LockAudioDevice" (audio_device_id @-> returning void)

let open_audio_device =
  foreign "SDL_OpenAudioDevice"
    (string_opt @-> bool @-> ptr audio_spec @-> ptr audio_spec @->
     Audio.allow @-> returning uint32_t)

let open_audio_device dev capture desired allow =
  let desiredc = audio_spec_to_c desired in
  let obtained = make audio_spec in
  match open_audio_device dev capture (addr desiredc) (addr obtained) allow
  with
  | id when id = Unsigned.UInt32.zero -> error ()
  | id -> Ok (id,  audio_spec_of_c obtained desired.as_ba_kind)

let pause_audio_device =
  foreign "SDL_PauseAudioDevice" (audio_device_id @-> bool @-> returning void)

let unlock_audio_device =
  foreign "SDL_UnlockAudioDevice" (audio_device_id @-> returning void)

(* Timer *)

let delay =
  foreign ~release_runtime_lock:true "SDL_Delay" (int32_t @-> returning void)

let get_ticks =
  foreign "SDL_GetTicks" (void @-> returning int32_t)

let get_performance_counter =
  foreign "SDL_GetPerformanceCounter" (void @-> returning int64_t)

let get_performance_frequency =
  foreign "SDL_GetPerformanceFrequency" (void @-> returning int64_t)

(* Platform and CPU information *)

let get_platform =
  foreign "SDL_GetPlatform" (void @-> returning string)

let get_cpu_cache_line_size =
  foreign "SDL_GetCPUCacheLineSize" (void @-> returning nat_to_ok)

let get_cpu_count =
  foreign "SDL_GetCPUCount" (void @-> returning int)

let get_system_ram =
  foreign "SDL_GetSystemRAM" (void @-> returning int)

let has_3d_now =
  foreign "SDL_Has3DNow" (void @-> returning bool)

let has_altivec =
  foreign "SDL_HasAltiVec" (void @-> returning bool)

let has_avx =
  foreign ~stub "SDL_HasAVX" (void @-> returning bool)

let has_mmx =
  foreign "SDL_HasMMX" (void @-> returning bool)

let has_rdtsc =
  foreign "SDL_HasRDTSC" (void @-> returning bool)

let has_sse =
  foreign "SDL_HasSSE" (void @-> returning bool)

let has_sse2 =
  foreign "SDL_HasSSE2" (void @-> returning bool)

let has_sse3 =
  foreign "SDL_HasSSE3" (void @-> returning bool)

let has_sse41 =
  foreign "SDL_HasSSE41" (void @-> returning bool)

let has_sse42 =
  foreign "SDL_HasSSE42" (void @-> returning bool)

(* Power management *)

type power_state =
  [ `Unknown | `On_battery | `No_battery | `Charging | `Charged ]

let power_state =
  [ sdl_powerstate_unknown, `Unknown;
    sdl_powerstate_on_battery, `On_battery;
    sdl_powerstate_no_battery, `No_battery;
    sdl_powerstate_charging, `Charging;
    sdl_powerstate_charged, `Charged; ]

type power_info =
  { pi_state : power_state;
    pi_secs : int option;
    pi_pct : int option; }

let get_power_info =
  foreign "SDL_GetPowerInfo" ((ptr int) @-> (ptr int) @-> returning int)

let get_power_info () =
  let secs = allocate int 0 in
  let pct = allocate int 0 in
  let s = get_power_info secs pct in
  let pi_state = try List.assoc s power_state with Not_found -> assert false in
  let pi_secs = match !@ secs with -1 -> None | secs -> Some secs in
  let pi_pct = match !@ pct with -1 -> None | pct -> Some pct in
  { pi_state; pi_secs; pi_pct }

end

(*---------------------------------------------------------------------------
   Copyright (c) 2013 Daniel C. Bünzli

   Permission to use, copy, modify, and/or distribute this software for any
   purpose with or without fee is hereby granted, provided that the above
   copyright notice and this permission notice appear in all copies.

   THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
   WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
   MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
   ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
   WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
   ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
   OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
  ---------------------------------------------------------------------------*)

end
