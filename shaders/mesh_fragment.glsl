precision mediump float;
varying vec3 vColor;
varying vec3 vNormal;
varying vec3 vPosition3D;  // xyz coordinates

uniform float hueMin;
uniform float hueMax;
uniform float chromaMin;
uniform float chromaMax;
uniform float valueMin;
uniform float valueMax;
uniform float useLighting;
uniform float showOutsideRGB;

void main() {
    bool positionInRange;
   
    if(vPosition3D.x == 0.0 && vPosition3D.z == 0.0) {
        positionInRange = true;  // always include grayscale (center axis)
    } else {
        // calculate angle from 3D position
        float angle = atan(vPosition3D.z, vPosition3D.x);
        if(angle < 0.0) angle += 2.0 * 3.14159;  // Normalize to [0, 2π]
       
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

    vec3 finalColor;
    if(useLighting > 0.5) {
        // diffuse lighting
        vec3 lightDirection = normalize(vec3(1.0, 1.0, 1.0));
        float lightIntensity = max(dot(normalize(vNormal), lightDirection), 0.3);
        finalColor = vColor * lightIntensity;
    } else {
        finalColor = vColor;
    }
    gl_FragColor = vec4(finalColor, 1.0);
}