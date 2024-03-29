# libtf

This library performs non-blocking inference on tensorflow frozen model. Based on the Tensorflow C API.

## Dependency

Install libtensorflow from https://www.tensorflow.org/install/lang_c

## Install

```bash
npm i libtf
```

## Usage

```js
var TFModel = require('libtf')


var model = TFModel() // To be called for each desired Tensorflow Session

// With options:
// var model = TFModel({
//   'allow_growth': false,       // If true, overrides gpu_memory_fraction
//   'gpu_memory_fraction': 1.0,
// })

;(async () => {

  await model.load('./model.pb') // Session is created and model loaded
  var input = {
    "isTrainingflag": false,  // Boolean
    "dropout_keep_prob": {    // Scalar
      "dim": [1],
      "data": new Float32Array([1]),
    },
    "inputs/enc_in": {        // Tensor, data must be of type Float32Array or UInt8Array
      "dim": [1,32],
      "data": new Float32Array(1*32),
    },
  }
  var result = await model.execute(input, ["dense_1/Softmax"]) // arg1 is input, arg2 is an array of output names

  console.log(result)

// Output:
// { 'dense_1/Softmax': // Output name
//   { dim: [ 1, 62 ],
//      data:
//       Float32Array [ ... ] 
//   } 
// }


})()
.catch(function (err) {
  console.log(err)
})



```