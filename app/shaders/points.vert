attribute vec3 aPos;
attribute vec4 aColor;
attribute float aSize;
uniform mat4 uMVP;
uniform float uScale;
uniform vec3 uEye;
uniform float uDofRange;
varying vec4 vColor;
varying float vClipZ;
varying float vClipW;
varying float vWorldRadius;
varying float vDofFade;
void main() {
  gl_Position = uMVP * vec4(aPos, 1.0);
  gl_PointSize = aSize * uScale / gl_Position.w;

  // Pass clip-space depth components and view-space sphere radius for spherical depth
  vClipZ = gl_Position.z;
  vClipW = gl_Position.w;
  // View-space half-size of the point sprite (the "sphere radius"):
  // pointSize_pixels = aSize * uScale / w; pixel-to-view conversion at this depth
  // uses tan(fov/2) / (viewport_height/2). With uScale = viewport_height * 0.15
  // and fov = PI/4: radius = aSize * 0.15 * tan(PI/8) = aSize * 0.06213
  vWorldRadius = aSize * 0.06213;

  // Depth-based dimming: dot product of point direction with eye direction
  // Points facing the camera are bright, back-facing ones dim down
  vec3 toEye = normalize(uEye);
  vec3 pointDir = normalize(aPos);
  float facing = dot(pointDir, toEye);
  // Remap from [-1,1] to [0.08, 1.0] — back-facing nearly invisible
  float dim = smoothstep(-1.0, 0.6, facing) * 0.92 + 0.08;

  // Radial depth of field: center of screen is sharp, edges fade out
  vec2 ndc = gl_Position.xy / gl_Position.w; // normalized device coords [-1,1]
  float radialDist = length(ndc);             // distance from screen center
  // uDofRange controls how wide the sharp center region is (0 = tiny, 7 = everything sharp)
  float dofFade = 1.0 - smoothstep(uDofRange * 0.2, uDofRange * 0.2 + 1.0, radialDist);
  dofFade = max(dofFade, 0.12); // never fully invisible

  // Blur factor: 0 = fully sharp, 1 = max blur
  float blur = 1.0 - dofFade;
  vDofFade = blur;

  // Scale up point size for out-of-focus nodes so the soft falloff has room
  gl_PointSize *= 1.0 + blur * 0.6;

  // aColor.a > 1.0 signals hover: excess is the hover factor (0→1)
  float hover = clamp(aColor.a - 1.0, 0.0, 1.0);
  float baseA = min(aColor.a, 1.0);
  float dimmedAlpha = baseA * dim * dofFade;
  // On hover, lerp alpha toward 1.0 (fully opaque, no dim/dof fade)
  // RGB stays untouched — fading is purely via alpha, not darkening to black
  vColor = vec4(aColor.rgb, mix(dimmedAlpha, 1.0, hover));
  // Also cancel blur on hover so hovered nodes stay crisp
  vDofFade = mix(blur, 0.0, hover);
}
