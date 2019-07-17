# node-tfjs-retrain

Retraining image classification model using TensorFlow.js in Node.

Based on the following js.tensorflow.org tutorials:

-   [Webcam Transfer learning](https://js.tensorflow.org/tutorials/webcam-transfer-learning.html)
-   [Training MNIST with Node.js](https://github.com/tensorflow/tfjs-examples/tree/master/mnist-node)

Excellent article from James Thomas that helped (_skip the stuff about downloading the model shards manually; there's a much simpler way described in the comments_):  
[Machine Learning in Node.js with TensorFlow.js](http://jamesthom.as/blog/2018/08/07/machine-learning-in-node-dot-js-with-tensorflow-dot-js/)

## Example Usages

-   Retrain and test model:  
    `node app.js --images_dir="C:/Retraining_Project/Images" --model_dir="C:/Retraining_Project/Model"`
-   Skip retraining; just test model:  
    `node app.js --images_dir="C:/Retraining_Project/Images" --model_dir="C:/Retraining_Project/Model" --skip_training=true`
-   Create sample images:  
    `node create_images.js C:/Retraining_Project/Images`


_ 	Full example with tensorboard:

		cli commands to start tensorboard (localhost:6006)

		start virtual environment -

    	source ./venv/bin/activate

 		start tensorboard on localhost:6006 -

    	(venv) $ tensorboard --logdir /tmp/tf_fit_logs

 		node-tfjs-retrain cli -
 
    	node app.js --images_dir="/Users/claymation296/dev/magic/synthetic" --model_dir="/Users/claymation296/dev/nathanML/ml-lock/src/model" --sizing="contain" --epochs="500"

 		shutdown virtual environment

    	(venv) $ deactivate



- 	app.js args:

		--batch_size_fraction -> Number default 0.2
		--dense_units 				-> Number default 100
		--epochs 							-> Number default 50
		--learning_rate 			-> Number default 0.0001
		--sizing 							-> String default 'cover'
		--logdir 							-> String default '/tmp/tf_fit_logs', tensorboard --logdir /tmp/tf_fit_logs


- 	augment_images.js args:

		--flip_images 					-> Boolean default false
		--adjust_brightness 		-> Boolean default false
		--ignore_subdirectories -> Boolean default false
		--labels_to_skip 				-> String default ''




## Help

I'm quite new to Node and TF, so there's a lot of room for cleanup and code improvements. My initial goal was to get something working and worry about minor improvements later. Feel free to submit a PR :)

# Initial TODOs:

-   [x] Create a script to automatically build a training data set to enable users to immediately get started
-   [x] Save trained model in the project directory
-   [x] Replace ui_mock.js with a TrainingParams class
