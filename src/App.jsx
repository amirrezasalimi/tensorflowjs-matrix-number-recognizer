import * as tf from '@tensorflow/tfjs'

import { signal } from '@preact/signals-react'
import './App.css'
import { useRef, useState } from 'react';
import clsx from 'clsx';
import { useLocalStorage } from 'usehooks-ts'
import CanvasMatrix from './components/canvas-matrix';
import madedDataset from "./dataset.json"
import { toast } from 'react-toastify';
const matrixItemsInRow = 8
const matrix = signal([...new Array(matrixItemsInRow * matrixItemsInRow)].map(() => 0));



function App() {
  const isDrag = useRef(false);
  const [label, setLabel] = useState("");
  const [dataset, saveDataset] = useLocalStorage("matrix-dataset", madedDataset);

  const [predictedClass, setPredictedClass] = useState(null);
  const modelRef = useRef(tf.sequential())
  const model = modelRef.current;

  const onMouseOverItem = (e) => {
    if (isDrag.current) {
      const i = e.target.dataset.id;
      matrix.value[i] = 1
      matrix.value = [...matrix.value]
    }
  }
  const onToggle = (e) => {
    if (!isDrag.current) {
      const i = e.target.dataset.id;
      matrix.value[i] = 0
      matrix.value = [...matrix.value]
    }

  }
  const onMouseUp = () => {
    const isEmpty = model.layers.length === 0;
    if (!isEmpty) {
      const inputTensor = tf.tensor2d([matrix.value]); // Convert the 1D array into a 2D tensor
      const prediction = model.predict(inputTensor);
      const predictedClass = tf.argMax(prediction, 1);
      predictedClass.array().then(array => {
        console.log('Predicted class:', array);
        toast.info(`Predicted: ${array[0]}`)
        setPredictedClass(`${array[0]}`)
      });
    }

    isDrag.current = false
  }

  // save test data
  const clear = () => {
    matrix.value = [...new Array(matrixItemsInRow * matrixItemsInRow)].map(() => 0)
  }
  const remove = (label, i) => {
    saveDataset({
      ...dataset,
      [label]: [...dataset[label]].reverse().filter((_, _i) => _i !== i)
    })
  }
  const save = () => {
    const sample = [...matrix.value]
    const name = label

    if (!dataset[name]) {
      saveDataset({
        ...dataset,
        [name]: [sample]
      })
    } else {
      saveDataset({
        ...dataset,
        [name]: [
          ...dataset[name],
          sample
        ]
      })
    }
    clear();
    // setLabel("");
  }

  // train stuff
  const train = async () => {

    let X = [];
    let y = [];

    for (let label in dataset) {
      for (let sample of dataset[label]) {
        X.push(sample);
        y.push(Number(label));
      }
    }


    model.add(tf.layers.dense({ units: 128, activation: 'relu', inputShape: [64] }));
    model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));

    model.compile({
      optimizer: 'adam',
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy'],
    });

    console.log('Model summary:');
    model.summary();


    // Convert the arrays into tensors
    const X_tensor = tf.tensor2d(X);
    const y_tensor = tf.oneHot(tf.tensor1d(y, 'int32'), 10); // Assuming 10 classes (0-9)

    const splitRatio = 0.85;
    const numSamples = X_tensor.shape[0];
    const numTrainSamples = Math.floor(numSamples * splitRatio);

    const [X_train, X_val] = tf.split(X_tensor, [numTrainSamples, numSamples - numTrainSamples], 0);
    const [y_train, y_val] = tf.split(y_tensor, [numTrainSamples, numSamples - numTrainSamples], 0);


    const batchSize = 32;
    const epochs = 30;


    const trainingToastId = toast.info("training...", {
      autoClose: false,
      progress: false
    })
    await model.fit(X_train, y_train, {
      batchSize: batchSize,
      epochs: epochs,
      validationData: [X_val, y_val],
    }).then(info => {
      toast.done(trainingToastId)
      toast.success("training finished!")
      console.log('Training finished, final accuracy:', info.history.acc[info.history.acc.length - 1]);
    }).catch(err => {
      console.log('Error during training:', err);
    });

  }
  return (
    <>
      <section className="main">
        <h1>
          Tensorflow.js 8x8 matrix number recognizer
        </h1>
        <span>
          there is few sample of each number in below dataset , just click on train and then, write your number in matrix
        </span>
        <div className="matrix"
          onMouseDown={(e) => {
            if (!isDrag.current) {
              isDrag.current = true;
              onMouseOverItem(e)
            }
          }}
          onMouseMove={onMouseOverItem}
          onMouseUp={onMouseUp}
        >
          {
            matrix.value.map((_, i) =>
              <div
                onClick={onToggle}
                data-id={i}
                className={clsx('item', matrix.value[i] == 1 && "active")} key={i}></div>
            )
          }
        </div>
        <button className='btn' onClick={clear} >
          clear
        </button>
        {predictedClass && <div>
          predicted: {predictedClass}
        </div>}
        <hr />

        <div className='break' />
        <div className='save-box'>
          <input placeholder='label...' type='text' value={label} onInput={(e) => {
            setLabel(e.target.value)
          }} />
          <button className='btn' onClick={save} >
            save
          </button>
        </div>

        <div className="dataset">
          <h3>
            dataset
          </h3>
          <div>
            <button className='btn blue' onClick={train} >
              train
            </button>
          </div>

          {Object.keys(dataset).reverse().map(num => {
            const items = [...dataset[num]].reverse();
            return <div className='item' key={num}>
              <h2>
                {num}
              </h2>
              <span>
                <b>{items.length}</b> items
              </span>
              <div className="matrix-items">
                {items.map((item, i) => <div className='matrix-canvas-item' key={i}>
                  <span className='delete' onClick={() => remove(num, i)}>x</span>
                  <CanvasMatrix matrix={item} width={64} height={64} rows={8} />
                </div>)}
              </div>
            </div>
          })}
        </div>
      </section>

    </>
  )
}

export default App
