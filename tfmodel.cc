#include <fstream>
#include <utility>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <napi.h>

#include "promiseWorker.hpp"

#include "tfmodel.h"

Napi::FunctionReference TFModel::constructor;

Napi::Object TFModel::Init(Napi::Env env, Napi::Object exports) {
  Napi::HandleScope scope(env);

  Napi::Function func = DefineClass(env, "TFModel", {
      InstanceMethod("load", &TFModel::load),
      InstanceMethod("execute", &TFModel::execute)
  });

  constructor = Napi::Persistent(func);
  constructor.SuppressDestruct();

  exports.Set("TFModel", func);
  return exports;
}

TFModel::TFModel(const Napi::CallbackInfo& info) : Napi::ObjectWrap<TFModel>(info) {
  Napi::Env env = info.Env();
  Napi::HandleScope scope(env);

  Napi::Object options;

  if(info[0].IsObject()) {
    options = info[0].As<Napi::Object>();
  } else {
    options = Napi::Object::New(env);
  }

  allow_growth = false;
  if(options.HasOwnProperty("allow_growth")) {
    allow_growth = options.Get("allow_growth").As<Napi::Boolean>();
  }
  gpu_memory_fraction = 1.0;
  if(options.HasOwnProperty("gpu_memory_fraction")) {
    gpu_memory_fraction = options.Get("gpu_memory_fraction").As<Napi::Number>();
  }

};

TFModel::~TFModel() {
  if(tf_sess) {
    TF_Status* s = TF_NewStatus();
    TF_CloseSession(tf_sess,s);
    TF_DeleteSession(tf_sess,s);
    TF_DeleteGraph(tf_graph);
    TF_DeleteStatus(s);  
  }
}

Napi::Object TFModel::NewInstance(Napi::Env env, Napi::Value arg) {
  Napi::EscapableHandleScope scope(env);
  Napi::Object obj = constructor.New({ arg });
  return scope.Escape(napi_value(obj)).ToObject();
}

typedef struct {
  std::string name;
  std::vector<int64_t> dim;
  uint32_t datasize;
  bool isBool;
  bool b_data;
  bool isFloat32Array;
  std::vector<float> fa_data;
  bool isUInt8Array;
  std::vector<unsigned char> u8a_data;
} TFInput;

Napi::Value TFModel::execute(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  //Napi::HandleScope scope(env);
  
  std::vector<TFInput> *inputs = new std::vector<TFInput>; // <-- to cleanup

  Napi::Object jsinputs = info[0].As<Napi::Object>();

  Napi::Array keys = jsinputs.GetPropertyNames();
  int len = ((Napi::Value)keys["length"]).As<Napi::Number>().Int32Value();
  for(int i = 0; i < len; i++) {
    std::string key = ((Napi::Value)keys[i]).As<Napi::String>();
    Napi::Value val = jsinputs.Get(key);

    if(val.IsBoolean()) {

      Napi::Boolean input = val.As<Napi::Boolean>();
      uint32_t datasize = sizeof(bool);

      (*inputs).push_back({ std::string(key), std::vector<int64_t>(), datasize, true, input, false, std::vector<float>(), false, std::vector<unsigned char>() });

    } else {

      Napi::Object input = val.As<Napi::Object>();
      Napi::Array arrdim = input.Get("dim").As<Napi::Array>();
      Napi::TypedArray arrdata = input.Get("data").As<Napi::TypedArray>();

      int dimlen = ((Napi::Value)arrdim["length"]).As<Napi::Number>().Int32Value();
      std::vector<int64_t> dim;

      uint32_t dataCnt = 1;
      for(int j = 0; j < dimlen; j++) {
        int64_t d = arrdim.Get(j).As<Napi::Number>().Int64Value();
        dim.push_back(d);
        dataCnt *= d;
      }
      //uint32_t datasize = dataCnt*sizeof(float);
      //(*inputs).push_back({ std::string(key), dim, datasize, false, false, true, std::vector<float>(jsdata.Data(), jsdata.Data()+dataCnt) });

      if(arrdata.TypedArrayType() == napi_typedarray_type::napi_uint8_array) {
        uint32_t datasize = dataCnt*sizeof(unsigned char);
        Napi::Uint8Array jsdata_uint8 = arrdata.As<Napi::Uint8Array>();
        (*inputs).push_back({ std::string(key), dim, datasize, false, false, false, std::vector<float>(), true, std::vector<unsigned char>(jsdata_uint8.Data(), jsdata_uint8.Data()+dataCnt) });
      } else {
        uint32_t datasize = dataCnt*sizeof(float);
        Napi::Float32Array jsdata_float = arrdata.As<Napi::Float32Array>();
        (*inputs).push_back({ std::string(key), dim, datasize, false, false, true, std::vector<float>(jsdata_float.Data(), jsdata_float.Data()+dataCnt), false, std::vector<unsigned char>() });
      }

    }

  }

  //

  std::vector<TF_Output> *output_names = new std::vector<TF_Output>; // <-- to cleanup
  std::vector<std::string> *output_names_str = new std::vector<std::string>; // <-- to cleanup

  Napi::Array outputs = info[1].As<Napi::Array>();
  int outlen = ((Napi::Value)outputs["length"]).As<Napi::Number>().Int32Value();
  for(int i = 0; i < outlen; i++) {
    std::string outname = std::string(outputs.Get(i).As<Napi::String>());
    output_names_str->push_back(outname);
    output_names->push_back({ TF_GraphOperationByName(tf_graph, outname.c_str()), 0 });
  }

  std::vector<TF_Tensor*> *output_values = new std::vector<TF_Tensor*>(output_names->size(), nullptr); // <-- to cleanup

  //

  PromiseWorker* wk = new PromiseWorker(env,
  [=] () {

    std::vector<TF_Output> input_names;
    std::vector<TF_Tensor*> input_values;

    for(uint32_t i = 0; i < (*inputs).size(); i++) {
      input_names.push_back({TF_GraphOperationByName(tf_graph, (*inputs)[i].name.c_str()), 0});
      if((*inputs)[i].isBool) {
        input_values.push_back(TF_NewTensor(TF_BOOL, nullptr, 0, &(*inputs)[i].b_data, (*inputs)[i].datasize, [] (void* data, size_t len, void* arg) {}, nullptr));  
      } else {
        //input_values.push_back(TF_NewTensor(TF_FLOAT, (*inputs)[i].dim.data(), (*inputs)[i].dim.size(), (*inputs)[i].fa_data.data(), (*inputs)[i].datasize, [] (void* data, size_t len, void* arg) {}, nullptr));
        if((*inputs)[i].isUInt8Array) {
          input_values.push_back(TF_NewTensor(TF_UINT8, (*inputs)[i].dim.data(), (*inputs)[i].dim.size(), (*inputs)[i].u8a_data.data(), (*inputs)[i].datasize, [] (void* data, size_t len, void* arg) {}, nullptr));
        } else {
          input_values.push_back(TF_NewTensor(TF_FLOAT, (*inputs)[i].dim.data(), (*inputs)[i].dim.size(), (*inputs)[i].fa_data.data(), (*inputs)[i].datasize, [] (void* data, size_t len, void* arg) {}, nullptr));
        }
      }
    }

    TF_Status * s = TF_NewStatus();

    TF_SessionRun(tf_sess,nullptr,input_names.data(),input_values.data(),input_names.size(),
        (*output_names).data(),(*output_values).data(),(*output_names).size(),
        nullptr,0,nullptr,s);
    bool failed = (TF_GetCode(s) != TF_OK);
    TF_DeleteStatus(s);

    for(uint32_t i = 0; i < input_values.size(); i++) {
      TF_DeleteTensor(input_values[i]);
    }

    if (failed) {
      throw std::runtime_error("TF_SessionRun failed!\n Error: " + std::string(TF_Message(s)));
    }

  },
  [=] (Napi::Env env) -> Napi::Value {

    Napi::Object retval = Napi::Object::New(env);

    for(uint32_t i = 0; i < output_values->size(); i++) {
      Napi::Array arrdim = Napi::Array::New(env);
      uint64_t datasize = 1;
      for(int j = 0; j < TF_NumDims((*output_values)[i]); j++) {
        uint64_t d = TF_Dim((*output_values)[i],j);
        arrdim.Set(j, Napi::Number::New(env, d));
        datasize *= d;
      }
      Napi::Object outobj = Napi::Object::New(env);
      outobj.Set("dim", arrdim);
      Napi::Float32Array data = Napi::Float32Array::New(env, datasize);
      const float * out_data = (const float *)TF_TensorData((*output_values)[i]);
      std::copy(out_data, out_data + datasize, data.Data());
      outobj.Set("data", data);

      retval.Set((*output_names_str)[i], outobj);

      TF_DeleteTensor((*output_values)[i]);
    }

    //cleanup
    delete inputs;
    delete output_names;
    delete output_names_str;
    delete output_values;

    return retval;
  });
  wk->Queue();
  return wk->Deferred().Promise();

  return env.Undefined();
}


Napi::Value TFModel::load(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  std::string fname = std::string(info[0].As<Napi::String>());

  PromiseWorker* wk = new PromiseWorker(env, 
  [=] () {

	  TF_Status* s = TF_NewStatus();
	  tf_graph = TF_NewGraph();

    //Read model file
    std::ifstream fs(fname, std::ios::binary | std::ios::in);
    
    if (!fs.good()) {
      throw std::runtime_error("File not found: " + fname);
    }

    fs.seekg(0, std::ios::end);
    int fsize=fs.tellg();

    fs.seekg(0, std::ios::beg);
    model_buf.resize(fsize);
    fs.read(model_buf.data(),fsize);

    fs.close();
    //

    TF_Buffer graph_def = {model_buf.data(), model_buf.size(), nullptr};

    TF_ImportGraphDefOptions* import_opts = TF_NewImportGraphDefOptions();
    TF_ImportGraphDefOptionsSetPrefix(import_opts, "");
    TF_GraphImportGraphDef(tf_graph, &graph_def, import_opts, s);

    if (TF_GetCode(s) != TF_OK) {
      throw std::runtime_error("load graph failed!\n Error: " + std::string(TF_Message(s)));
    }

    TF_SessionOptions* sess_opts = TF_NewSessionOptions();
    //
    if(allow_growth) {
      uint8_t config[4] = {0x32, 0x2, 0x20, 0x1}; //config.gpu_options.allow_growth = True
      TF_SetConfig(sess_opts, (void*)config, 4, s);
    } else {
      if(gpu_memory_fraction > 0.0 && gpu_memory_fraction < 1.0) {
        uint8_t config[11] = {0x32, 0x9, 0x9, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xf0, 0x3f}; //config.gpu_options.per_process_gpu_memory_fraction = 1.0
        for(int i = 0; i < 8; i++) {
          config[3+i] = ((unsigned char *)(&gpu_memory_fraction))[i];
        }
        TF_SetConfig(sess_opts, (void*)config, 11, s);
      }
      // else default config
    }
    if (TF_GetCode(s) != TF_OK) {
      throw std::runtime_error("TF_NewSession failed!\n Error: " + std::string(TF_Message(s)));
    }
    //
    tf_sess = TF_NewSession(tf_graph, sess_opts, s);
    if (TF_GetCode(s) != TF_OK) {
      throw std::runtime_error("TF_NewSession failed!\n Error: " + std::string(TF_Message(s)));
    }

    TF_DeleteStatus(s);
  },
  nullptr);
  wk->Queue();
  return wk->Deferred().Promise();

}
