uniform sampler3D interiorTexture;
varying vec3 vObjectPosition;
varying vec3 vNormal;
uniform float useLighting;

void main() {
    // use a rectangular texture like it's cylindrical coordinates
    // x = radius (0 is central pole)
    // y = angular coordinate, [0,1] maps to [0, 2pi]
    // z unchanged
    
    // convert y to theta
    float theta = vObjectPosition.y * 2.0 * 3.14159265359;
    
    // Ccnvert cylindrical to cartesian for texture lookup
    // x is the radius, theta determines the angular position
    float x = vObjectPosition.x;
    float texX = x * cos(theta);
    float texY = x * sin(theta);
    float texZ = vObjectPosition.z;
    
    // remap from [-1,1] back to [0,1] for texture sampling
    vec3 texCoords = vec3(
        texX * 0.5 + 0.5,
        texY * 0.5 + 0.5,
        texZ
    );
    
    // sample with transformed coordinates
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