uniform sampler3D interiorTexture;
varying vec3 vObjectPosition;
varying vec3 vNormal;
uniform float useLighting;

void main() {
    // convert xz location to polar coordinates
    vec2 xzPos = vObjectPosition.xz - vec2(0.5, 0.5);
    float theta = atan(xzPos.y, xzPos.x) + 3.14159265359;
    float radius = length(xzPos) * 2.0;
    // y remains vertical coordinate for cylindrical coordinates
    float height = vObjectPosition.y;
    
    
    // remap from [-1,1] back to [0,1] for texture sampling
    vec3 texCoords = vec3(
        radius, 
        height, 
        theta / (2.0 * 3.14159265359)
    );
    
    vec4 texColor = texture(interiorTexture, texCoords);
    
    // DEBUG: visualize texture coords as white to black
    // radial
    // texColor = vec4(radius, radius, radius, 1.0);
    // angular
    // texColor = vec4(theta / (2.0 * 3.14159265359), theta / (2.0 * 3.14159265359), theta / (2.0 * 3.14159265359), 1.0);
    
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