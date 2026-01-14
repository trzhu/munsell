uniform sampler3D interiorTexture;
varying vec3 vObjectPosition;
varying vec3 vColor;
varying vec3 vNormal;
varying vec3 vPosition3D;

void main() {
    vec4 worldPosition = modelMatrix * vec4(position, 1.0);

    vNormal = normalize(normalMatrix * normal);

    vec3 normalizedPos;
    // x from -38 to +38 -> 0 to 1
    normalizedPos.x = (worldPosition.x + 38.0) / 76.0;
    // y from 0 to 40 -> 0 to 1
    normalizedPos.y = worldPosition.y / 40.0;
    // z from -38 to +38 -> 0 to 1 
    normalizedPos.z = (worldPosition.z + 38.0) / 76.0;

    vObjectPosition = clamp(normalizedPos, 0.0, 1.0);

    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}