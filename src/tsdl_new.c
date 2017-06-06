#include <caml/alloc.h>
#include <caml/custom.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/fail.h>

#include <SDL2/SDL.h>

CAMLprim value TSDL_GL_SetAttribute(value a, value v) {
  CAMLparam2(a, v);
  CAMLreturn(Val_int(SDL_GL_SetAttribute(Int_val(a), Int_val(v))));
}

CAMLprim value TSDL_CreateWindow_native(value title, value x, value y, value w, value h, value flags) {
  CAMLparam5(title, x, y, w, h);
  CAMLxparam1(flags);
  CAMLlocal1(win_ret);
  SDL_Window *window = SDL_CreateWindow(String_val(title), Int_val(x), Int_val(y), Int_val(w), Int_val(h), Int_val(flags));
  win_ret = caml_alloc_small(1, Abstract_tag);
  Field(win_ret, 0) = (long)window;
  CAMLreturn(win_ret);
}

CAMLprim value TSDL_CreateWindow_bytecode( value * argv, int argn ) {
  return TSDL_CreateWindow_native(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5]);
}

void TSDL_DestroyWindow(value window) {
  CAMLparam1(window);
  SDL_DestroyWindow((SDL_Window *)Field(window, 0));
  CAMLreturn0;
}

CAMLprim value TSDL_GetWindowSize(value window) {
  CAMLparam1(window);
  CAMLlocal1(ret);
  int w;
  int h;
  SDL_GetWindowSize((SDL_Window *)Field(window, 0), &w, &h);
  ret = caml_alloc_small(2, 0);
  Field(ret, 0) = Val_int(w);
  Field(ret, 1) = Val_int(w);
  CAMLreturn(ret);
}

void TSDL_SetWindowSize(value window, value w, value h) {
  CAMLparam3(window, w, h);
  SDL_SetWindowSize((SDL_Window *)Field(window, 0), Int_val(w), Int_val(h));
  CAMLreturn0;
}

CAMLprim value TSDL_Init(value flags) {
  CAMLparam1(flags);
  CAMLreturn(Val_int(SDL_Init(Int_val(flags))));
}

CAMLprim value TSDL_GL_CreateContext(value window) {
  CAMLparam1(window);
  CAMLlocal1(ret);
  ret = caml_alloc_small(1, Abstract_tag);
  Field(ret, 0) = (long)SDL_GL_CreateContext((SDL_Window *)Field(window, 0));
  CAMLreturn(ret);
}

CAMLprim value TSDL_GL_MakeCurrent(value window, value context) {
  CAMLparam2(window, context);
  CAMLreturn(Val_int(SDL_GL_MakeCurrent((SDL_Window *)Field(window, 0), (SDL_GLContext)Field(context, 0))));
}

CAMLprim value TSDL_PollEvent() {
  CAMLparam0();
  CAMLlocal2(ret, wrapped);
  SDL_Event *event = malloc(sizeof(SDL_Event));
  int eventAvailable = SDL_PollEvent(event);
  if (eventAvailable == 0) {
    CAMLreturn(Val_int(0));
  }
  ret = caml_alloc_small(1, Abstract_tag);
  Field(ret, 0) = (long)event;

  wrapped = caml_alloc_small(1, 0);
  Field(wrapped, 0) = ret;
  CAMLreturn(wrapped);
}

CAMLprim value T_get_type(value event) {
  CAMLparam1(event);
  SDL_Event *e = (SDL_Event *)Field(event, 0);
  CAMLreturn(Val_int(e->type));
}

CAMLprim value T_get_mouse_button_button(value event) {
  CAMLparam1(event);
  SDL_Event *e = (SDL_Event *)Field(event, 0);
  CAMLreturn(Val_int(e->button.button));
}

CAMLprim value T_get_mouse_button_x(value event) {
  CAMLparam1(event);
  SDL_Event *e = (SDL_Event *)Field(event, 0);
  CAMLreturn(Val_int(e->button.x));
}

CAMLprim value T_get_mouse_button_y(value event) {
  CAMLparam1(event);
  SDL_Event *e = (SDL_Event *)Field(event, 0);
  CAMLreturn(Val_int(e->button.y));
}

CAMLprim value T_get_mouse_motion_x(value event) {
  CAMLparam1(event);
  SDL_Event *e = (SDL_Event *)Field(event, 0);
  CAMLreturn(Val_int(e->motion.xrel));
}

CAMLprim value T_get_mouse_motion_y(value event) {
  CAMLparam1(event);
  SDL_Event *e = (SDL_Event *)Field(event, 0);
  CAMLreturn(Val_int(e->motion.yrel));
}

CAMLprim value T_get_window_event(value event) {
  CAMLparam1(event);
  SDL_Event *e = (SDL_Event *)Field(event, 0);
  CAMLreturn(Val_int(e->window.event));
  // switch (e->window.event) {
  //   case SDL_WINDOWEVENT_SHOWN:
  //       CAMLreturn(hash_variant("Shown"));
  //   case SDL_WINDOWEVENT_HIDDEN:
  //       CAMLreturn(hash_variant("Hidden"));
  //   case SDL_WINDOWEVENT_EXPOSED:
  //       CAMLreturn(hash_variant("Exposed"));
  //   case SDL_WINDOWEVENT_MOVED:
  //       CAMLreturn(hash_variant("Moved"));
  //   case SDL_WINDOWEVENT_RESIZED:
  //       CAMLreturn(hash_variant("Resized"));
  //   case SDL_WINDOWEVENT_SIZE_CHANGED:
  //       CAMLreturn(hash_variant("Size_changed"));
  //   case SDL_WINDOWEVENT_MINIMIZED:
  //       CAMLreturn(hash_variant("Minimized"));
  //   case SDL_WINDOWEVENT_MAXIMIZED:
  //       CAMLreturn(hash_variant("Maximized"));
  //   case SDL_WINDOWEVENT_RESTORED:
  //       CAMLreturn(hash_variant("Restored"));
  //   case SDL_WINDOWEVENT_ENTER:
  //       CAMLreturn(hash_variant("Enter"));
  //   case SDL_WINDOWEVENT_LEAVE:
  //       CAMLreturn(hash_variant("Leave"));
  //   case SDL_WINDOWEVENT_FOCUS_GAINED:
  //       CAMLreturn(hash_variant("Focus_gained"));
  //   case SDL_WINDOWEVENT_FOCUS_LOST:
  //       CAMLreturn(hash_variant("Focus_lost"));
  //   case SDL_WINDOWEVENT_CLOSE:
  //       CAMLreturn(hash_variant("Close"));
  //   default:
  //   // HEEH
  //     CAMLreturn(Val_int(0));
  //       // SDL_Log("Window %d got unknown event %d",
  //       //         event->window.windowID, event->window.event);
  //       // break;
  //   }
}

CAMLprim value T_get_keyboard_repeat(value event) {
  CAMLparam1(event);
  SDL_Event *e = (SDL_Event *)Field(event, 0);
  CAMLreturn(Val_int(e->key.repeat));
}

CAMLprim value T_get_keyboard_keycode(value event) {
  CAMLparam1(event);
  SDL_Event *e = (SDL_Event *)Field(event, 0);
  CAMLreturn(Val_int(e->key.keysym.sym));
}

CAMLprim value TSDL_GetPerformanceCounter() {
  CAMLparam0();
  CAMLreturn(caml_copy_int64(SDL_GetPerformanceCounter()));
}

CAMLprim value TSDL_GetPerformanceFrequency() {
  CAMLparam0();
  CAMLreturn(caml_copy_int64(SDL_GetPerformanceFrequency()));
}

void TSDL_GL_SwapWindow(value window) {
  CAMLparam1(window);
  SDL_GL_SwapWindow((SDL_Window *)Field(window, 0));
  CAMLreturn0;
}

void TSDL_Delay(value delay) {
  CAMLparam1(delay);
  SDL_Delay(Int_val(delay));
  CAMLreturn0;
}

void TSDL_Quit() {
  CAMLparam0();
  SDL_Quit();
  CAMLreturn0;
}

CAMLprim value TSDL_GetWindowSurface(value window) {
  CAMLparam1(window);
  CAMLlocal1(ret);
  ret = caml_alloc_small(1, Abstract_tag);
  SDL_Surface *s = SDL_GetWindowSurface((SDL_Window *)Field(window, 0));
  Field(ret, 0) = (long)s;
  CAMLreturn(ret);
}

CAMLprim value TSDL_LoadBMP(value bmp_name) {
  CAMLparam1(bmp_name);
  CAMLlocal1(ret);
  ret = caml_alloc_small(1, Abstract_tag);
  Field(ret, 0) = (long)SDL_LoadBMP(String_val(bmp_name));
  CAMLreturn(ret);
}

CAMLprim value TSDL_BlitSurface(value surf1, value surf2) {
  CAMLparam2(surf1, surf2);
  int error = SDL_BlitSurface((SDL_Surface *)Field(surf1, 0), NULL, (SDL_Surface *)Field(surf2, 0), NULL);
  CAMLreturn(Val_int(error));
}

CAMLprim value TSDL_UpdateWindowSurface(value window) {
  CAMLparam1(window);
  int error = SDL_UpdateWindowSurface((SDL_Window *)Field(window, 0));
  CAMLreturn(Val_int(error));
}

CAMLprim value TSDL_GetError() {
  CAMLparam0();
  CAMLlocal1(ret);
  ret = caml_copy_string(SDL_GetError());
  CAMLreturn(ret);
}

CAMLprim value T_or(value a, value b) {
  CAMLparam2(a, b);
  CAMLreturn(Val_int(Int_val(a) | Int_val(b)));
}
