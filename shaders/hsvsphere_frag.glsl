varying vec3 vNormal;

vec3 hsv2rgb(vec3 c) {
  vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
  vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
  return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() {
  float h = (atan(vNormal.z, vNormal.x) / (2.0 * 3.1415926)) + 0.5;
  float s = acos(vNormal.y) / 3.1415926;
  float v = 1.0;

  vec3 hsv = vec3(h, s, v);
  vec3 rgb = hsv2rgb(hsv);

  gl_FragColor = vec4(rgb, 1.0);
}
