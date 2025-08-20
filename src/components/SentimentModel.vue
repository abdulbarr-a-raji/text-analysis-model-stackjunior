<template>
  <div class="hello">
    <h1>{{ msg }}</h1>
    <p v-if="loadingEncoder" class="txt-small">
      <strong>Loading Universal Sentence Encoder...</strong>
    </p>

    <div v-else>
      <button id="train-button" v-on:click="this.run">Train model</button>
      <br>
      <button id="predict-button" v-on:click="this.getPrediction">Predict with model</button>
    </div>
    
    <p v-if="trainingComplete">
      Accuracy = {{ finalAcc }}%
    </p>
    <p v-if="startedPrediction" class="txt-small">
      <strong>Current test data: </strong>
      <pre>{{ chosenTestQuestion }}</pre>
      <br>
      Model predicts... {{ modelPrediction }}
    </p>
  </div>
</template>

<script>
/*
plan:
load json data (maybe add options for user to add their own json files for training)
convert to embeddings
create and compile model
train model
don't forget disposals
allow for predictions
*/

export default {
  name: 'SentimentModel',
  props: {
    msg: String,
    datasetFile: String,
    classes: Object
  },
  async mounted() {
    this.tf = window.tf;
    await this.loadEmbedder();
  },
  data() {
    return {
      EMBEDDINGS_SIZE: undefined,
      embedder: null,
      model: null,
      dataset: {},
      chosenTestQuestion: "",
      modelPrediction: "N/A",
      finalAcc: undefined,

      //flags
      loadingEncoder: false,
      startedPrediction: false,
      trainingComplete: false
    }
  },
  methods: {
    async loadEmbedder() {
      try {
        if(this.embedder) {
          console.log('USE model already loaded...');
        } else {
          this.loadingEncoder = true;
          let start = performance.now();
          this.embedder = await window.use.load();
          let end = performance.now();
          this.logTimeTaken(start, end, "Loading Tensorflow USE..."); // mean value is ~2.2 seconds
          this.memoryMeter();
          console.log("Universal Sentence Encoder loaded?", this.embedder ? "yes" : "no");
          this.loadingEncoder = false;
        } // this if statement is not mitigating increase in number of tensors for each mount, must find solution using dispose()
      } catch (err) {
        console.error('Failed to load USE model:', err);
      }
    },
    async getEmbeddingsAndLabels(json_rel_file_path) {
      //load json file, not loading leading '{' char, leave for now
      console.log("json file at:", json_rel_file_path);
      try {
        const response = await fetch(json_rel_file_path);
        const fetch_contents = await response.text();
        console.log(fetch_contents.slice(1, 20));

        // jsonDataset = await response.json().catch(() => {});
        // because response.json() omits the leading '{' for some reason, the above
        // line of code doesn't work
        this.dataset = JSON.parse(fetch_contents)
        console.log("file found: ", response.status, response.ok);
        console.log(`Loaded '${json_rel_file_path}' file`);
      } catch (e) {
        console.error(`Failed to load a supposed '${json_rel_file_path}' file`, e);
      }
      console.log("data?", this.dataset);
      
      const text_inputs = [], labels = [];
      this.dataset.training.forEach((item) => {
        text_inputs.push(item.text.trim().toLowerCase());
        labels.push(this.classes[item.label]);
      });

      this.tf.util.shuffleCombo(
        text_inputs,
        labels
      );

      //preparing validation data
      const val_text = [], val_labels = [];
      this.dataset.validation.forEach((item) => {
        val_text.push(item.text.trim().toLowerCase());
        val_labels.push(this.classes[item.label]);
      });

      return {
        train: [
        await this.embedder.embed(text_inputs), 
        this.tf.tidy(() => {
          return this.tf.oneHot(this.tf.tensor1d(labels, "int32"), 
          Object.values(this.classes).length)
        })
      ],
      val: [
        await this.embedder.embed(val_text), 
        this.tf.tidy(() => {
          return this.tf.oneHot(this.tf.tensor1d(val_labels, "int32"), 
          Object.values(this.classes).length)
        })
      ]};
    },
    async initialiseModel() {
      this.model = this.tf.sequential();

      // input layer
      this.model.add(
        this.tf.layers.dense({
          inputShape: [this.EMBEDDINGS_SIZE],
          units: 128,
          activation: "relu"
        })
      );
      // hidden layer
      this.model.add(
        this.tf.layers.dense({
          units: 64,
          activation: "relu"
        })
      );
      // output layer
      this.model.add(
        this.tf.layers.dense({
          units: Object.values(this.classes).length,
          activation: "softmax"
        })
      );

      // Compile with optimized settings
      const model_optimizer = this.tf.train.adam(); // lr values to try: 0.01, 0.001, 0.0001, and so on
      this.model.compile({
        optimizer: model_optimizer,
        loss: "categoricalCrossentropy",
        metrics: ["accuracy"],
      });

      this.memoryMeter();

      console.log("--> Classifcation model created successfully <--");
      this.model.summary();
    },
    async trainClassificationModel() {
      let start = performance.now();
      const datastructure = await this.getEmbeddingsAndLabels(this.datasetFile);
      const [Xtrain, Ytrain] = datastructure.train;
      let end = performance.now();
      this.logTimeTaken(start, end, "Fetching dataset as tensors...");
      this.memoryMeter();
      console.log("x size:", Xtrain.shape);
      Xtrain.print();

      this.EMBEDDINGS_SIZE = Xtrain.shape[1];
      const NUM_EPOCHS = 10;
      await this.initialiseModel();

      start = performance.now();
      const trainingResults = await this.model.fit(Xtrain, Ytrain, {
        shuffle: true,
        validationData: datastructure.val,
        batchSize: 16,
        epochs: NUM_EPOCHS
      });
      end = performance.now();
      this.logTimeTaken(start, end, "Training Classification model...");

      console.log("Training results:", trainingResults.history);
      this.trainingComplete = true;
      this.finalAcc = (trainingResults.history.acc[NUM_EPOCHS-1]*100).toFixed(2);
      this.memoryMeter();
      // Xtrain.dispose();
      // Ytrain.dispose();
    },
    memoryMeter() {
      console.log("memory meter (# of tensors):", this.tf.memory().numTensors);
    },
    logTimeTaken(start, end, title="") {
      const timeTakenInSeconds = ((end - start) / 1000).toFixed(2);
      console.log(
        `
        ${title}
        Time taken(s) = ${timeTakenInSeconds} seconds
        `
      );
    },
    getPrediction() {
      this.getRandomTestData();
      this.embedder.embed([this.chosenTestQuestion]).then((input_tns) => {
        const rawPrediction = this.model.predict(input_tns);
        rawPrediction.print();
        const predictionValue = rawPrediction.argMax(-1).dataSync()[0];

        this.startedPrediction = true;
        this.modelPrediction = Object.keys(this.classes)
                              .find(key => this.classes[key] === predictionValue);
      });
    },
    getRandomTestData() {
      const selected = Math.floor(Math.random() * this.dataset.testing.length);
      this.chosenTestQuestion = this.dataset.testing[selected].text.trim().toLowerCase();
    },
    async run() { // redundant?
      await this.trainClassificationModel();
    }
  }
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
h3 {
  margin: 40px 0 0;
}
ul {
  list-style-type: none;
  padding: 0;
}
li {
  display: inline-block;
  margin: 0 10px;
}
a {
  color: #42b983;
}
.txt-small {
  font-size: smaller;
}
</style>
