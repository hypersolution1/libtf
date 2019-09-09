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

;(async () => {

  await model.load('./model.pb') 
  var input = {
    "isTrainingflag": false,  // boolean
    "dropout_keep_prob": {    // scalar
      "dim": [1],
      "data": new Float32Array([1]),
    },
    "inputs/enc_in": {        // tensor, data must be of type Float32Array
      "dim": [1,32],
      "data": new Float32Array(1*32),
    },
  }
  var result = await model.execute(input, ["dense_1/Softmax"]) // input, array of output names

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