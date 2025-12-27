uniform sampler3D interiorTexture;
varying vec3 vObjectPosition;
varying vec3 vNormal;
uniform float useLighting;

void main() {
    // Sample the 3D texture using the normalized position coordinates
    vec4 texColor = texture(interiorTexture, vObjectPosition);
    
    // gl_FragColor = texColor;

    vec3 finalColor;
    if(useLighting > 0.5) {
        // diffuse lighting
        vec3 lightDirection = normalize(vec3(1.0, 1.0, 1.0));
        float lightIntensity = max(dot(normalize(vNormal), lightDirection), 0.3);
        finalColor = texColor.rgb * lightIntensity;
    } else {
        finalColor = texColor.rgb;
    }

    gl_FragColor = vec4(finalColor, 1.0);

    // debug - use position as colour
    // gl_FragColor = vec4(vObjectPosition, 1.0);
    
    // debug - translucent material
    // gl_FragColor = vec4(1.0, 1.0, 1.0, 0.5);
}