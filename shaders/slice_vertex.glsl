uniform sampler3D interiorTexture;
varying vec3 vObjectPosition;

void main() {
    vec4 worldPosition = modelMatrix * vec4(position, 1.0);

    vec3 normalizedPos;
    // x from -38 to +38 -> 0 to 1
    normalizedPos.x = (worldPosition.x + 38.0) / 76.0;
    // y from 0 to 30 -> 0 to 1
    normalizedPos.y = worldPosition.y / 30.0;
    // z from -38 to +38 -> 0 to 1 
    normalizedPos.z = (worldPosition.z + 38.0) / 76.0;

    vObjectPosition = clamp(normalizedPos, 0.0, 1.0);

    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}