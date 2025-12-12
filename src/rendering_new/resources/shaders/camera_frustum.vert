#version 430 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec2 aUV;

layout(location = 2) in mat4 aInstanceMatrix;
layout(location = 6) in vec4 aInstanceColorAlpha;
layout(location = 7) in uint aIsValidation;

uniform mat4 viewProj;
uniform vec3 viewPos;
uniform bool pickingMode = false;

out vec3 FragPos;
out vec4 vertexColor;
flat out int instanceID;
flat out uint isValidation;

void main() {
    instanceID = gl_InstanceID;
    vec4 worldPos = aInstanceMatrix * vec4(aPos, 1.0);

    isValidation = aIsValidation;
    FragPos = vec3(worldPos);
    gl_Position = viewProj * worldPos;

    if (pickingMode) {
        int id = gl_InstanceID + 1;
        vertexColor = vec4(
            float((id >> 16) & 0xFF) / 255.0,
            float((id >> 8) & 0xFF) / 255.0,
            float(id & 0xFF) / 255.0,
            1.0
        );
    } else {
        vertexColor = aInstanceColorAlpha;
    }
}
