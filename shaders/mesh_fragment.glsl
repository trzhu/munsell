precision mediump float;
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
  // Filter by ranges - discard fragments outside bounds (backup safety)
  bool hueInRange;
  if (hueMin <= hueMax) {
    hueInRange = (vHue >= hueMin && vHue <= hueMax);
  } else {
    hueInRange = (vHue >= hueMin || vHue <= hueMax);
  }
  
  if (!hueInRange ||
      vChroma < chromaMin || vChroma > chromaMax ||
      vValue < valueMin || vValue > valueMax) {
    discard;
  }
  
  // Simple diffuse lighting calculation
  vec3 lightDirection = normalize(vec3(1.0, 1.0, 1.0));
  float lightIntensity = max(dot(normalize(vNormal), lightDirection), 0.3);
  
  // Apply lighting to the vertex color from PLY
  vec3 litColor = vColor * lightIntensity;
  
  gl_FragColor = vec4(litColor, 1.0);
}