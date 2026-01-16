uniform sampler3D interiorTexture;
varying vec3 vObjectPosition;
varying vec3 vColor;
varying vec3 vNormal;
varying vec3 vPosition3D;

void main() {
    vNormal = normalize(normalMatrix * normal);

    vec3 normalizedPos;
    // x from -38 to +38 -> 0 to 1
    normalizedPos.x = (position.x + 38.0) / 76.0;
    // y from 0 to 40 -> 0 to 1
    normalizedPos.y = position.y / 40.0;
    // z from -38 to +38 -> 0 to 1 
    normalizedPos.z = (position.z + 38.0) / 76.0;

    vObjectPosition = clamp(normalizedPos, 0.0, 1.0);

    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}