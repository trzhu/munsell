precision mediump float;
attribute vec3 color;
// attribute vec3 position;
// attribute vec3 normal;
attribute float hue;
attribute float chroma;
attribute float value;
varying vec3 vColor;
varying vec3 vNormal;
varying float vHue;
varying float vChroma;
varying float vValue;
uniform float hueMin;
uniform float hueMax;
uniform float chromaMin;
uniform float chromaMax;
uniform float valueMin;
uniform float valueMax;

void main() {
  vColor = color;
  vNormal = normalize(normalMatrix * normal);
  vHue = hue;
  vChroma = chroma;
  vValue = value;
  
  // Filter by ranges - move vertices outside bounds far away
  bool hueInRange;
  if (hueMin <= hueMax) {
    // Normal case: hueMin = 30, hueMax = 90
    hueInRange = (vHue >= hueMin && vHue <= hueMax);
  } else {
    // Wraparound case: hueMin = 350, hueMax = 30
    hueInRange = (vHue >= hueMin || vHue <= hueMax);
  }
  
  bool inRange = hueInRange && 
                 vChroma >= chromaMin && vChroma <= chromaMax &&
                 vValue >= valueMin && vValue <= valueMax;
  
  vec4 worldPosition = modelViewMatrix * vec4(position, 1.0);
  
//   if (!inRange) {
//     // Move vertices outside bounds far away (behind camera)
//     worldPosition.z = -10000.0;
//   }
  
  gl_Position = projectionMatrix * worldPosition;
}