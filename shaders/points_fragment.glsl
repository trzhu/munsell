precision mediump float;
varying vec3 vColor;
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
    // make points circular instead of square
    vec2 coord = gl_PointCoord - vec2(0.5);
    if(length(coord) > 0.5) {
        discard;
    }

    // todo: fix value wraparound
    if(vHue < hueMin || vHue > hueMax ||
        vChroma < chromaMin || vChroma > chromaMax ||
        vValue < valueMin || vValue > valueMax) {
        discard;
    }

    // rgb colour from the frag file
    gl_FragColor = vec4(vColor, 1.0);
}