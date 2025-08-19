// import * as use from '@tensorflow-models/universal-sentence-encoder';

let embedder = null;

export default function getEmbedder() {
  if (!embedder) {
    embedder = window.use.load({
      modelUrl: 'https://storage.googleapis.com/tfjs-models/savedmodel/universal_sentence_encoder_lite/model.json'
    });
  }
  return embedder;
}
