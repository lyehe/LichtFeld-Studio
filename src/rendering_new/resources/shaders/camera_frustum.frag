#version 430 core

in vec3 FragPos;
in vec4 vertexColor;
flat in int instanceID;
flat in uint isValidation;

out vec4 FragColor;

uniform vec3 viewPos;
uniform int highlightIndex = -1;
uniform vec3 trainHighlightColor = vec3(1.0, 0.55, 0.0);
uniform vec3 valHighlightColor = vec3(0.9, 0.75, 0.0);
uniform bool pickingMode = false;
uniform float minimumPickDistance = 0.5;

void main() {
    if (pickingMode) {
        float distance = length(viewPos - FragPos);
        if (distance < minimumPickDistance) {
            discard;
        }
        FragColor = vertexColor;
        return;
    }

    vec4 finalColor = vertexColor;

    if (instanceID == highlightIndex) {
        finalColor.rgb = (isValidation > 0u) ? valHighlightColor : trainHighlightColor;
        finalColor.a = min(1.0, finalColor.a + 0.3);
    }

    FragColor = finalColor;
}
