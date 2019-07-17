//@ts-check
const fs    = require('fs');
const tf    = require('@tensorflow/tfjs-node-gpu');
const fg    = require('fast-glob');
const fse   = require('fs-extra');
const sharp = require('sharp');
const path  = require('path');

// We use a version of mobilenet which 
// expects input images to be 224px X 224px.
const IMAGE_SIZE = 224;

// // Save an image tensor to disk.
// const tensorToFile = (filename, tensor) => {
//   return new Promise((resolve, reject) => {
//     const png = new PNG({
//       width: tensor.shape[2],
//       height: tensor.shape[1]
//     });
//     png.data = tensor.dataSync();
//     png
//       .pack()
//       .pipe(fs.createWriteStream(filename))
//       .on('error', reject)
//       .on('close', resolve);
//   });
// };

// // Add an alpha channel to a 3 channel tensor.
// const getTensor4d = (data, info) => {     
//   const outShape = [1, info.height, info.width, info.channels]; 
//   return tf.tidy(() => {    
//     const tensor = tf.tensor4d(data, outShape, 'int32');
//     if (info.channels === 3) { // jpg/rgb
//       const withAlpha = tensor.pad(
//         [[null, null], [null, null], [null, null], [null, 1]], 
//         255 // full alpha
//       );
//       return withAlpha;
//     }
//     return tensor;
//   });
// };


// Remove png alpha channel since our models expect
// 3 channel shaped tensors.
const stripAlphaChannel = (tensor, info) => {
  return tf.tidy(() => tensor.slice(
    [0, 0, 0 ,0], 
    [1, info.height, info.width, 3]
  ));
};

// Accept rgb and rgba data, ie jpeg and png formats.
const imageToTensor = (pixelData, imageInfo) => {
  return tf.tidy(() => {
    // rgba -> rgb / the rest of the pipeline throws if 
    // alpha channel is present so pull it out and 
    // keep a shape [1, 224, 224, 3] tensor
    const outShape = [1, imageInfo.height, imageInfo.width, imageInfo.channels];
    const tensor   = tf.tensor4d(pixelData, outShape, 'int32'); 
    const noAlpha  = stripAlphaChannel(tensor, imageInfo);
    // Normalize the rgb data from [0, 255] to [-1, 1].
    const normalized = noAlpha.  
      toFloat().
      div(tf.scalar(127)).
      sub(tf.scalar(1));

    return normalized;
  });
};
// Resize/crop image from center without distortion
// to match dimentions expected by mobilenet. 
const fileToTensor = async (filename, sizing) => {
  const {data, info} = await sharp(filename).
    resize({
      fit:    sharp.fit[sizing], 
      height: IMAGE_SIZE, 
      width:  IMAGE_SIZE
    }).
    raw().
    toBuffer({resolveWithObject: true});

  return imageToTensor(data, info);
};


const getDirectories = imagesDirectory => fse.readdir(imagesDirectory);


const getImagesInDirectory = dir => fg([
  path.join(dir, '*.jpg'),
  path.join(dir, '*/*.jpg'),
  path.join(dir, '*.png'),
  path.join(dir, '*/*.png')
]);


const readImagesDirectory = async imagesDirectory => {
  const directories = await getDirectories(imagesDirectory);

  const getImagesPromises = directories.
    filter(dir => dir !== '.DS_Store'). // fix for apple directories
    map(async dir => {
      const p      = path.join(imagesDirectory, dir);
      const images = await getImagesInDirectory(p);
      return {label: dir, images};      
    });

  const result = await Promise.all(getImagesPromises);
  return result;
};


class Data {

  constructor() {
    this.dataset         = null;
    this.labelsAndImages = null;
    this.sizing          = 'cover';
  }


  getEmbeddingsForImage(index) {
    return this.dataset.images.gather([index]);
  }


  fileToTensor(filename) {
    return fileToTensor(filename, this.sizing);
  }


  // imageToTensor(image, numChannels) {
  //     return imageToTensor(image, numChannels);
  // }


  labelIndex(label) {
    return this.labelsAndImages.findIndex(item => 
      item.label === label);
  }


  async loadLabelsAndImages(imagesDirectory, sizing) {
    this.sizing = sizing;
    this.labelsAndImages = await readImagesDirectory(imagesDirectory);
  }


  async loadTrainingData(model) {
    const numClasses = this.labelsAndImages.length;
    const numImages = 
      this.labelsAndImages.reduce((acc, item) => 
        acc + item.images.length, 0);

    const embeddings = [];
    const labels     = new Int32Array(numImages);
    const embeddingsShape    = model.outputs[0].shape.slice(1);
    const embeddingsFlatSize = tf.util.sizeFromShape(embeddingsShape);
    embeddingsShape.unshift(numImages); // add number of images to front of shape

    console.log('Loading Training Data');
    console.time('Loading Training Data');
    // Loop through the files and populate the 'images' and 'labels' arrays
    let labelsOffset = 0;

    for (const element of this.labelsAndImages) {
      const labelIndex = this.labelIndex(element.label);
      for (const image of element.images) {
        const t = await fileToTensor(image, this.sizing);
        const prediction = model.predict(t);
        t.dispose();
        // Use tf.squeeze to emove the first dimention from prediction tensor.
        // From [1, 7 , 7, 1024] to [7, 7, 1024].
        // Later they will be stacked back together
        // to form a 4d tensor as input to custom model.
        embeddings.push(prediction.squeeze());
        labels.set([labelIndex], labelsOffset);
        labelsOffset += 1;
      }
      console.timeLog('Loading Training Data', {
        label: element.label,
        count: element.images.length
      });
    }
    // Combine all embedding tensors into a 4d tensor
    // where the first dimension is equal to the number
    // of images in the dataset. 
    const imagesTensor = tf.stack(embeddings);
    // Memory management 
    embeddings.forEach(tensor => {
      tensor.dispose();
    });
    this.dataset = {
      images: imagesTensor,
      labels: tf.oneHot(tf.tensor1d(labels, 'int32'), numClasses)
    };
  }


  // async loadTrainingData(model) {
  //     const numClasses = this.labelsAndImages.length;
  //     const numImages = this.labelsAndImages.reduce((accum, item) => 
  //                         accum + item.images.length, 0);

  //     const embeddingsShape = model.outputs[0].shape.slice(1);
  //     const embeddingsFlatSize = tf.util.sizeFromShape(embeddingsShape);
  //     embeddingsShape.unshift(numImages); // put numImages in front of shape array
  //     // prediction array size 50176 same as flatSize
  //     const totalDataSize = tf.util.sizeFromShape(embeddingsShape);
  //     // Maximum array length is 2^30 - 1
  //     const maxSize = (2 ** 30) - 1;

  //     const batchCount = totalDataSize > maxSize ? 
  //                          Math.ceil(totalDataSize / maxSize) : 
  //                          1;
  //     const maxBatchSize = totalDataSize / batchCount;
  //     const predictionsPerBatch = Math.floor(maxBatchSize / embeddingsFlatSize);

  //     const batchSize = predictionsPerBatch * embeddingsFlatSize;
  //     const embeddings = [new Float32Array(batchSize)];        
  //     const labels = new Int32Array(numImages);

  //     // Loop through the files and populate the 'images' and 'labels' arrays
  //     let batchIndex = 0;
  //     let embeddingsOffset = 0;
  //     let labelsOffset = 0;
  //     console.log('Loading Training Data');
  //     console.time('Loading Training Data');

  //     for (const element of this.labelsAndImages) {
  //         const labelIndex = this.labelIndex(element.label);
  //         for (const image of element.images) {
  //           const t = await fileToTensor(image, this.sizing);
  //           tf.tidy(() => {
  //             const prediction = model.predict(t);
  //             embeddings[batchIndex].set(prediction.dataSync(), embeddingsOffset);
  //             labels.set([labelIndex], labelsOffset);
  //           });
  //           t.dispose();
  //           embeddingsOffset += embeddingsFlatSize;
  //           if (embeddingsOffset >= batchSize) {
  //             batchIndex += 1;
  //             embeddingsOffset = 0;
  //             embeddings.push(new Float32Array(batchSize));
  //           }

  //           labelsOffset += 1;
  //         }
  //         console.timeLog('Loading Training Data', {
  //           label: element.label,
  //           count: element.images.length
  //         });
  //     }



  //     // TODO:
  //     //      batch data using args
  //     //      shuffle data
  //     //      seperate 15% for validation data



  //     // const shuffledIndices = new Int32Array(
  //     //   tf.util.createShuffledIndices(dataset.labels.shape[0])
  //     // );
  //     // dataset.images.gather(shuffledIndices)
  //     // embeddings[batchIndex].set(prediction.dataSync(), embeddingsOffset);
  //     // slice




  //     const datasetFromEmbeddings = (shape, labels) => embeddings => {
  //       function* dataGenerator() {            
  //         const numElements = embeddings.length;
  //         let index = 0;
  //         while (index < numElements) {
  //           const shuffledIndices = new Int32Array(
  //             tf.util.createShuffledIndices(labels.shape[0])
  //           );
  //           const imagesTensor = tf.tensor4d(embeddings[index], shape);
  //           yield {
  //             xs: imagesTensor.gather(shuffledIndices), 
  //             ys: labels.gather(shuffledIndices)
  //           };            
  //           index += 1;
  //         }
  //       }

  //       return tf.data.generator(dataGenerator);
  //     };
      

  //     const labelsTensor = tf.oneHot(tf.tensor1d(labels, 'int32'), numClasses);
  //     const getDataset   = datasetFromEmbeddings(embeddingsShape, labelsTensor);

  //     const trainingDataset   = getDataset(trainingEmbeddings);
  //     const validationDataset = getDataset(validationEmbeddings);

  //     this.dataset = {
  //       images:     trainingDataset,
  //       labels:     labelsTensor,
  //       shape:      embeddingsShape,
  //       validation: validationDataset
  //     };
  // }


  // async loadTrainingData(model) {
  //     const numClasses = this.labelsAndImages.length;
  //     const numImages = this.labelsAndImages.reduce(
  //         (acc, item) => acc + item.images.length,
  //         0
  //     );
  //     const embeddingsShape = model.outputs[0].shape.slice(1);
  //     const embeddingsFlatSize = tf.util.sizeFromShape(embeddingsShape);
  //     embeddingsShape.unshift(numImages);
  //     const embeddings = new Float32Array(
  //         tf.util.sizeFromShape(embeddingsShape)
  //     );
  //     const labels = new Int32Array(numImages);

  //     // Loop through the files and populate the 'images' and 'labels' arrays
  //     let embeddingsOffset = 0;
  //     let labelsOffset = 0;
  //     console.log("Loading Training Data");
  //     console.time("Loading Training Data");
  //     for (const element of this.labelsAndImages) {
  //         let labelIndex = this.labelIndex(element.label);
  //         for (const image of element.images) {
  //             let t = await fileToTensor(image, this.sizing);
  //             tf.tidy(() => {
  //                 let prediction = model.predict(t);
  //                 embeddings.set(prediction.dataSync(), embeddingsOffset);
  //                 labels.set([labelIndex], labelsOffset);
  //             });
  //             t.dispose();

  //             embeddingsOffset += embeddingsFlatSize;
  //             labelsOffset += 1;
  //         }
  //         console.timeLog("Loading Training Data", {
  //             label: element.label,
  //             count: element.images.length
  //         });
  //     }

  //     this.dataset = {
  //         images: tf.tensor4d(embeddings, embeddingsShape),
  //         labels: tf.oneHot(tf.tensor1d(labels, "int32"), numClasses)
  //     };
  // }
}

module.exports = new Data();
