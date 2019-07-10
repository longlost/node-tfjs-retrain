//@ts-check
const fs    = require('fs');
const tf    = require("@tensorflow/tfjs-node-gpu");
const fg    = require("fast-glob");
const fse   = require("fs-extra");
const sharp = require("sharp");
const path  = require("path");


// dont forget to remove opencv4nodejs


const IMAGE_SIZE = 224;
// const {PNG} = require('pngjs');
// sharp.cache(false);



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


// // add alpha channel
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


const stripAlphaChannel = (tensor, info) => {
  return tf.tidy(() => tensor.slice(
    [0, 0, 0 ,0], 
    [1, info.height, info.width, 3]
  ));
};

// accept rgb and rgba data
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


const fileToTensor = async (filename, sizing) => {
  const {data, info} = await sharp(filename).
    resize({
      fit: sharp.fit[sizing], 
      height: IMAGE_SIZE, 
      width: IMAGE_SIZE
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
      this.dataset = null;
      this.labelsAndImages = null;
      this.sizing = 'cover';
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
        return this.labelsAndImages.findIndex(item => item.label === label);
    }

    async loadLabelsAndImages(imagesDirectory, sizing) {
      this.sizing = sizing;
      this.labelsAndImages = await readImagesDirectory(imagesDirectory);
    }

    async loadTrainingData(model) {
        const numClasses = this.labelsAndImages.length;
        const numImages = this.labelsAndImages.reduce(
            (acc, item) => acc + item.images.length,
            0
        );

        const embeddingsShape = model.outputs[0].shape.slice(1);
        const embeddingsFlatSize = tf.util.sizeFromShape(embeddingsShape);
        embeddingsShape.unshift(numImages);
        const embeddings = new Float32Array(
            tf.util.sizeFromShape(embeddingsShape)
        );
        const labels = new Int32Array(numImages);

        // Loop through the files and populate the 'images' and 'labels' arrays
        let embeddingsOffset = 0;
        let labelsOffset = 0;
        console.log("Loading Training Data");
        console.time("Loading Training Data");
        for (const element of this.labelsAndImages) {
            let labelIndex = this.labelIndex(element.label);
            for (const image of element.images) {
                let t = await fileToTensor(image, this.sizing);
                tf.tidy(() => {
                    let prediction = model.predict(t);
                    embeddings.set(prediction.dataSync(), embeddingsOffset);
                    labels.set([labelIndex], labelsOffset);
                });
                t.dispose();

                embeddingsOffset += embeddingsFlatSize;
                labelsOffset += 1;
            }
            console.timeLog("Loading Training Data", {
                label: element.label,
                count: element.images.length
            });
        }

        this.dataset = {
            images: tf.tensor4d(embeddings, embeddingsShape),
            labels: tf.oneHot(tf.tensor1d(labels, "int32"), numClasses)
        };
    }
}

module.exports = new Data();
