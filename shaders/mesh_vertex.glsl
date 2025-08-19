precision mediump float;

attribute vec3 color;
attribute float hue;
attribute float chroma;
attribute float value;

varying vec3 vColor;
varying vec3 vNormal;
varying vec3 vPosition3D;

void main() {
    vColor = color;
    vNormal = normalize(normalMatrix * normal);
    
    // Calculate 3D coordinates from hue/chroma/value
    float hueRadians = hue * 3.14159 / 180.0;  // Convert degrees to radians
    vPosition3D = vec3(
        chroma * cos(hueRadians),
        value,
        chroma * sin(hueRadians)
    );
    
    vec4 worldPosition = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * worldPosition;
}