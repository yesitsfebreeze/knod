attribute vec3 aPos;
attribute vec4 aColor;
uniform mat4 uMVP;
uniform vec3 uEye;
uniform float uDofRange;
varying vec4 vColor;
void main() {
  gl_Position = uMVP * vec4(aPos, 1.0);

  // Depth dimming matching the point shader
  vec3 toEye = normalize(uEye);
  float pLen = length(aPos);
  vec3 pointDir = pLen > 0.0 ? aPos / pLen : vec3(0.0);
  float facing = dot(pointDir, toEye);
  float dim = smoothstep(-1.0, 0.6, facing) * 0.92 + 0.08;

  // Radial depth of field: center of screen is sharp, edges fade out
  vec2 ndc = gl_Position.xy / gl_Position.w;
  float radialDist = length(ndc);
  float dofFade = 1.0 - smoothstep(uDofRange * 0.2, uDofRange * 0.2 + 1.0, radialDist);
  dofFade = max(dofFade, 0.12);

  // aColor.a > 1.0 signals hover: excess is the hover factor (0→1)
  float hover = clamp(aColor.a - 1.0, 0.0, 1.0);
  float baseA = min(aColor.a, 1.0);
  float dimmedAlpha = baseA * dim * dofFade;
  // Darken RGB for depth — thin lines need color darkening, not just alpha fade
  float rgbDim = mix(dim * dofFade, 1.0, hover);
  vColor = vec4(aColor.rgb * rgbDim, mix(dimmedAlpha, baseA, hover));
}
