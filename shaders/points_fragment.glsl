precision mediump float;
varying vec3 vColor;
varying vec3 vPosition3D;
varying float vIsClipped;
uniform float hueMin; // in radians
uniform float hueMax;
uniform float chromaMin;
uniform float chromaMax;
uniform float valueMin;
uniform float valueMax;
uniform float showOutsideRGB;

void main() {
    // make points circular instead of square
    vec2 coord = gl_PointCoord - vec2(0.5);
    if(length(coord) > 0.5) {
        discard;
    }
    
    //uUse 3D position logic like the mesh shader
    bool positionInRange;
    
    if(vPosition3D.x == 0.0 && vPosition3D.z == 0.0) {
        positionInRange = true;  // always include grayscale
    } else {
        float angle = atan(vPosition3D.z, vPosition3D.x);
        if(angle < 0.0) angle += 2.0 * 3.14159;
        
        if(hueMin < hueMax) {
            positionInRange = (angle >= hueMin && angle <= hueMax);
        } else {
            positionInRange = (angle >= hueMin || angle <= hueMax);
        }
    }
    
    if(!positionInRange ||
        length(vec2(vPosition3D.x, vPosition3D.z)) < chromaMin ||
        length(vec2(vPosition3D.x, vPosition3D.z)) > chromaMax ||
        vPosition3D.y < valueMin || vPosition3D.y > valueMax) {
        discard;
    }
    
    if(vIsClipped > 0.5 && showOutsideRGB < 0.5) {
        discard;
    }
    
    gl_FragColor = vec4(vColor, 1.0);
}