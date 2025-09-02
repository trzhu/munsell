uniform sampler3D interiorTexture;
varying vec3 vObjectPosition;

void main() {
    // Sample the 3D texture using the normalized position coordinates
    vec4 texColor = texture(interiorTexture, vObjectPosition);
    
    gl_FragColor = texColor;

    // debug - use position as colour
    // gl_FragColor = vec4(vObjectPosition, 1.0);
    
    // debug - translucent material
    // gl_FragColor = vec4(1.0, 1.0, 1.0, 0.5);
}