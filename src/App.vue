<template>
  <div id="app">
    <SentimentModel 
      msg="Sentiment Analysis"
      datasetFile="/storage/sentiments_data.json"
      :classes="{negative:0, neutral:1, positive:2}"
    />
    <br>
    <hr>
    <br>
    <SentimentModel 
      msg="Subject Analysis"
      datasetFile="/storage/subjects_data.json"
      :classes="{english:0, maths:1, science:2}"
    />
  </div>
</template>

<script>
import SentimentModel from './components/SentimentModel.vue'

export default {
  name: 'App',
  components: {
    SentimentModel
  }
}
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 60px;
}
</style>
