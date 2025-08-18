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

    // Handle hue wraparound
    bool hueInRange;
    if(hueMin < hueMax) {
        hueInRange = (vHue >= hueMin && vHue <= hueMax);
    } else {
    // hue wraparound when hueMin > huemax
        hueInRange = (vHue >= hueMin || vHue <= hueMax);
    }

    if(!hueInRange ||
        vChroma < chromaMin || vChroma > chromaMax ||
        vValue < valueMin || vValue > valueMax) {
        discard;
    }

    // rgb colour from the ply file
    gl_FragColor = vec4(vColor, 1.0);
}