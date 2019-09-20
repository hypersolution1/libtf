#ifndef TFMODEL_H
#define TFMODEL_H

#include <napi.h>
#include "tensorflow/c/c_api.h"

class TFModel : public Napi::ObjectWrap<TFModel> {
  public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports);
    static Napi::Object NewInstance(Napi::Env env, Napi::Value arg);
    TFModel(const Napi::CallbackInfo& info);
    ~TFModel();

  private:
    static Napi::FunctionReference constructor;
    Napi::Value load(const Napi::CallbackInfo& info);
    Napi::Value execute(const Napi::CallbackInfo& info);

    std::vector<char> model_buf;
    TF_Session * tf_sess;
    TF_Graph * tf_graph;
    bool allow_growth;
    double gpu_memory_fraction;
};

#endif
