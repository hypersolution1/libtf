#include <napi.h>
#include "tfmodel.h"

Napi::Object CreateObject(const Napi::CallbackInfo& info) {
  return TFModel::NewInstance(info.Env(), info[0]);
}

Napi::Object InitAll(Napi::Env env, Napi::Object exports) {
  Napi::Object new_exports = Napi::Function::New(env, CreateObject, "CreateObject");
  return TFModel::Init(env, new_exports);
}

NODE_API_MODULE(addon, InitAll)
