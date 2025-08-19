attribute vec3 color;
attribute float hue;
attribute float chroma;
attribute float value;
attribute float isClipped;
varying vec3 vColor;
varying vec3 vPosition3D;  // Changed from individual vHue, vChroma, vValue
varying float vIsClipped;
uniform float uSize;

void main() {
    vColor = color;
    vIsClipped = isClipped;
    
    // calculate 3D coordinates from hue/chroma/value
    float hueRadians = hue * 3.14159 / 180.0;
    vPosition3D = vec3(
        chroma * cos(hueRadians),
        value,
        chroma * sin(hueRadians)
    );
    
    gl_PointSize = uSize;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}