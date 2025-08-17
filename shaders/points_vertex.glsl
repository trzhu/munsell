attribute vec3 color;
attribute float hue;
attribute float chroma;
attribute float value;
varying vec3 vColor;
varying float vHue;
varying float vChroma;
varying float vValue;
uniform float uSize;

void main() {
  vColor = color;
  vHue = hue;
  vChroma = chroma;
  vValue = value;
  gl_PointSize = uSize;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}