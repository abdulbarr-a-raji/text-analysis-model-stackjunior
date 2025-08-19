<template>
  <div class="hello">
    <h1>{{ msg }}</h1>
    <p>
      Placeholder
    </p>
    <button id="dev-button" v-on:click="this.run">Run in console</button>
    <pre>{{ EMBEDDINGS_SIZE }}</pre>
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
// import getEmbedder from '/public/loadEmbedder.js'

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
      embedder: null,
      EMBEDDINGS_SIZE: undefined,
      model: null,
      tempSentimentDataset: {
        "training": [
          {"text": "I had an amazing time with my friends at the park today.","label": "positive"},
          {"text": "This coffee tastes incredible and really brightened my morning.","label": "positive"},
          {"text": "The new software update makes my laptop run so smoothly.","label": "positive"},
          {"text": "I feel so grateful for the support of my family.","label": "positive"},
          {"text": "The movie last night was absolutely fantastic and uplifting.","label": "positive"},
          {"text": "Our team worked really well together and achieved great results.","label": "positive"},
          {"text": "I love how peaceful and relaxing the weather is today.","label": "positive"},
          {"text": "The book I’m reading is inspiring and full of hope.","label": "positive"},
          {"text": "My workout session left me energized and happy.","label": "positive"},
          {"text": "That concert was one of the best experiences of my life.","label": "positive"},
          {"text": "The food at that restaurant was delicious and satisfying.","label": "positive"},
          {"text": "I feel confident and ready to take on new challenges.","label": "positive"},
          {"text": "My pet always makes me smile with its playful nature.","label": "positive"},
          {"text": "Traveling to new places gives me so much joy and excitement.","label": "positive"},
          {"text": "Learning something new today made me feel accomplished.","label": "positive"},
          {"text": "The flowers in the garden are blooming beautifully.","label": "positive"},
          {"text": "I’m so proud of the progress I’ve made this week.","label": "positive"},
          {"text": "Helping others always brings me a sense of fulfillment.","label": "positive"},
          {"text": "The project presentation went really well and was appreciated.","label": "positive"},
          {"text": "Waking up early gave me such a productive and positive day.","label": "positive"}
        ],
        "testing": [
          {"text": "Music always lifts my spirits and keeps me motivated.","label": "positive"},
          {"text": "I’m looking forward to tomorrow with excitement and optimism.","label": "positive"}
        ]
      },
      tempSubjectDataset: undefined
    }
  },
  methods: {
    async loadEmbedder() {
      try {
        if(this.embedder) {
          console.log('USE model already loaded:\n', this.embedder);
        } else {
          let start = performance.now();
          this.embedder = await window.use.load();
          let end = performance.now();
          this.logTimeTaken(start, end); // mean value is ~2.2 seconds
          this.memoryMeter();
          console.log("Universal Sentence Embedder?", this.embedder ? "yes" : "no");
          console.log('USE model loaded', this.embedder);
        } // this if statement is not mitigating increase in number of tensors for each mount, must find solution using dispose()
      } catch (err) {
        console.error('Failed to load USE model:', err);
      }
    },
    async getEmbeddingsAndLabels(json_rel_file_path) {
      //load json file, not loading leading '{' char, leave for now
      let jsonDataset = {};
      console.log("json file at:", json_rel_file_path);
      try {
        const response = await fetch(json_rel_file_path);
        const fetch_contents = await response.text();
        console.log(fetch_contents.slice(1, 50));
        jsonDataset = await response.json().catch(() => {
          console.log("file found: ", response.status, response.ok);
          console.log(`Failed to load a supposed '${json_rel_file_path}' file`);
        });
      } catch (e) {
        console.error(e);
      }
      
      const text_inputs = [], labels = [];
      console.log("data?", jsonDataset);
      console.log("data?", this.tempSentimentDataset);
      this.tempSentimentDataset.training.forEach((item) => {
        text_inputs.push(item.text.trim().toLowerCase());
        labels.push(this.classes[item.label]);
      });
      console.log("text data?", text_inputs);
      console.log("labels?", labels);

      this.tf.util.shuffleCombo(
        text_inputs,
        labels
      );

      // const embeddings = ;

      return [
        await this.embedder.embed(text_inputs), 
        this.tf.tidy(() => {
          return this.tf.oneHot(this.tf.tensor1d(labels, "int32"), 
          Object.values(this.classes).length)
        })
      ];
    },
    async initialiseModel() {
      this.model = this.tf.sequential();

      // input layer
      this.model.add(
        this.tf.layers.dense({
          inputShape: [this.EMBEDDINGS_SIZE],
          units: 64,
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
      const [Xtrain, Ytrain] = await this.getEmbeddingsAndLabels(this.datasetFile);
      let end = performance.now();
      this.logTimeTaken(start, end);
      this.memoryMeter();
      console.log("x size:", Xtrain.shape);
      Xtrain.print();

      this.EMBEDDINGS_SIZE = Xtrain.shape[1];
      console.log("embedding size:", this.EMBEDDINGS_SIZE);
      await this.initialiseModel();
      console.log("input shape:", this.model.inputShape);

      let trainingResults = await this.model.fit(Xtrain, Ytrain, {
        shuffle: true,
        // batchSize: 16,
        epochs: 10
      });

      console.log("Training results:", trainingResults.history);
      this.memoryMeter();
      // Xtrain.dispose();
      // Ytrain.dispose();
    },
    memoryMeter() {
      console.log("memory meter (# of tensors):", this.tf.memory().numTensors);
    },
    logTimeTaken(start, end) {
      const timeTakenInSeconds = ((end - start) / 1000).toFixed(2);
      console.log("Time taken(s) = "+timeTakenInSeconds, "seconds");
    },
    async run() {
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
</style>
