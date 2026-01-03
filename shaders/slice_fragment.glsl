uniform sampler3D interiorTexture;
varying vec3 vObjectPosition;
varying vec3 vNormal;
uniform float useLighting;

void main() {
    // convert xyz to cylindrical
    float radius = vObjectPosition.x;
    float theta = mod(vObjectPosition.z + 0.75, 1.0) * 2.0 * 3.14159265359;
    
    // read in cylindrical coordinates
    float texX = radius * cos(theta);
    float texY = vObjectPosition.y; // y = height stays the same
    float texZ = radius * sin(theta);
    
    // remap from [-1,1] back to [0,1] for texture sampling
    vec3 texCoords = vec3(
        texX * 0.5 + 0.5, texY, texZ * 0.5 + 0.5
    );
    
    vec4 texColor = texture(interiorTexture, texCoords);
    
    vec3 finalColor;
    if(useLighting > 0.5) {
        vec3 lightDirection = normalize(vec3(1.0, 1.0, 1.0));
        float lightIntensity = max(dot(normalize(vNormal), lightDirection), 0.3);
        finalColor = texColor.rgb * lightIntensity;
    } else {
        finalColor = texColor.rgb;
    }
    
    gl_FragColor = vec4(finalColor, 1.0);
}